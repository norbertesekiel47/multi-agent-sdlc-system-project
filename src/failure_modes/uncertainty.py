"""Uncertainty escalation — two deterministic trigger paths, first-trigger-wins.

Architecture §2.9:

Path A (validation path):
  Pydantic structured-output parsing fails 3 times in a row for
  the same agent on the same step → trigger ``pydantic_validation_3x``.

Path B (external-signal path):
  One of these deterministic conditions in a single task:
  - Persistent test failure across 3 retries → ``persistent_test_failure``
  - Reviewer rejects the same fix twice (same diff_hash) → ``same_fix_rejected_twice``
  - Tool-error rate >50% in a 10-call window → ``tool_error_rate_exceeded``

LLM self-confidence is explicitly NOT used as a trigger
(VAL-UNCERTAINTY-006, VAL-UNCERTAINTY-010).

First-fired trigger wins (VAL-UNCERTAINTY-005): only one
uncertainty_escalation outcome per task is recorded.

When a trigger fires: log to Langfuse, write ``outcomes`` row,
raise a LangGraph ``interrupt()`` with the trigger reason.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Number of consecutive Pydantic failures before escalation
_PYDANTIC_FAIL_THRESHOLD: int = 3

# Number of persistent test failures before escalation
_PERSISTENT_TEST_FAILURE_THRESHOLD: int = 3

# Tool-error rate window size
_TOOL_ERROR_WINDOW_SIZE: int = 10

# Tool-error rate threshold (fraction)
_TOOL_ERROR_RATE_THRESHOLD: float = 0.5

# Trigger type literals
type UncertaintyTriggerName = Literal[
    "pydantic_validation_3x",
    "persistent_test_failure",
    "same_fix_rejected_twice",
    "tool_error_rate_exceeded",
]


class UncertaintyTrigger:
    """A single uncertainty escalation trigger result.

    Carries the trigger name and relevant detail for Langfuse,
    outcomes row, and HITL escalation.
    """

    def __init__(
        self,
        *,
        trigger: UncertaintyTriggerName,
        agent_name: str,
        step_index: int,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.trigger = trigger
        self.agent_name = agent_name
        self.step_index = step_index
        self.detail = detail or {}

    def to_outcome_data(self) -> dict[str, Any]:
        """Return a dict for writing an ``outcomes`` row."""
        return {
            "outcome": "uncertainty_escalation",
            "detail": {
                "trigger": self.trigger,
                "agent_name": self.agent_name,
                "step_index": self.step_index,
                **self.detail,
            },
        }

    def to_hitl_details(self) -> dict[str, Any]:
        """Return HITL escalation details for the interrupt payload."""
        return {
            "cause": "uncertainty_escalation",
            "trigger": self.trigger,
            "agent_name": self.agent_name,
            "step_index": self.step_index,
            "explanation": _TRIGGER_EXPLANATIONS.get(
                self.trigger, "Uncertainty escalation triggered."
            ),
            **self.detail,
        }


_TRIGGER_EXPLANATIONS: dict[str, str] = {
    "pydantic_validation_3x": (
        "Agent output failed Pydantic validation 3 times in a row, "
        "suggesting the LLM is unable to produce valid structured output."
    ),
    "persistent_test_failure": (
        "Tests have failed across 3 consecutive QA retries, "
        "suggesting the code changes are not resolving the issue."
    ),
    "same_fix_rejected_twice": (
        "The Reviewer rejected the same diff_hash twice, "
        "suggesting the Coder is not producing meaningfully different fixes."
    ),
    "tool_error_rate_exceeded": (
        "More than 50% of the last 10 tool calls returned errors, "
        "suggesting systemic issues with the tool execution environment."
    ),
}


class UncertaintyEscalation:
    """Tracks uncertainty escalation conditions per task.

    Path A: Pydantic validation failure tracking
    ----------------------------------------------
    Maintains a per-(agent, step) counter of consecutive Pydantic
    validation failures.  When it reaches 3, the ``pydantic_validation_3x``
    trigger fires.  A successful validation resets the counter
    (VAL-UNCERTAINTY-008).

    Path B: External signal tracking
    ---------------------------------
    - ``persistent_test_failure``: counts consecutive QA retries
      where ``TestReport.failed > 0``.  Fires at 3.
    - ``same_fix_rejected_twice``: tracks diff_hashes that have been
      rejected by the Reviewer.  Fires when the same hash is rejected
      a second time.
    - ``tool_error_rate_exceeded``: maintains a sliding window of the
      last 10 tool call outcomes (success/error) per agent.  Fires when
      errors exceed 50% of the window.

    First-trigger-wins (VAL-UNCERTAINTY-005): once any trigger fires
    for a task, no further uncertainty_escalation outcomes are produced.

    LLM self-confidence is never used (VAL-UNCERTAINTY-006):
    no code path inspects an LLM self-reported confidence field.
    """

    def __init__(self) -> None:
        # Path A: Pydantic validation failure counters per (agent, step)
        self._pydantic_fail_counters: dict[str, int] = {}

        # Path B: persistent test failure counter
        self._test_failure_count: int = 0

        # Path B: rejected diff_hashes — maps hash → rejection count
        self._rejected_diff_hashes: dict[str, int] = {}

        # Path B: tool error sliding windows per agent
        self._tool_error_windows: dict[str, deque[bool]] = {}

        # First-trigger-wins flag
        self._trigger_fired: bool = False

    # ── Path A: Pydantic validation failure tracking ────────────

    def record_pydantic_failure(
        self, agent_name: str, step_index: int
    ) -> UncertaintyTrigger | None:
        """Record a Pydantic validation failure for (agent, step).

        Returns an UncertaintyTrigger if the consecutive failure count
        reaches 3, or None otherwise.

        VAL-UNCERTAINTY-001: Three consecutive Pydantic failures
        escalate with trigger ``pydantic_validation_3x``.
        """
        if self._trigger_fired:
            return None

        key = f"{agent_name}:{step_index}"
        current = self._pydantic_fail_counters.get(key, 0)
        new_count = current + 1
        self._pydantic_fail_counters[key] = new_count

        logger.debug(
            "Pydantic failure recorded: %s → %d/%d",
            key, new_count, _PYDANTIC_FAIL_THRESHOLD,
        )

        if new_count >= _PYDANTIC_FAIL_THRESHOLD:
            self._trigger_fired = True
            logger.warning(
                "Uncertainty escalation: pydantic_validation_3x for %s",
                key,
            )
            return UncertaintyTrigger(
                trigger="pydantic_validation_3x",
                agent_name=agent_name,
                step_index=step_index,
                detail={"consecutive_failures": new_count},
            )
        return None

    def record_pydantic_success(self, agent_name: str, step_index: int) -> None:
        """Record a successful Pydantic validation, resetting the counter.

        VAL-UNCERTAINTY-008: A successful Pydantic-valid response
        resets the consecutive-failure counter to 0.
        """
        key = f"{agent_name}:{step_index}"
        if key in self._pydantic_fail_counters:
            logger.debug(
                "Pydantic success: resetting counter for %s (was %d)",
                key, self._pydantic_fail_counters[key],
            )
        self._pydantic_fail_counters.pop(key, None)

    # ── Path B: Persistent test failure tracking ────────────────

    def record_test_failure(
        self, agent_name: str, step_index: int
    ) -> UncertaintyTrigger | None:
        """Record a test failure (TestReport.failed > 0).

        Returns an UncertaintyTrigger if 3 consecutive test failures
        occur, or None otherwise.

        VAL-UNCERTAINTY-002: Persistent test failure across 3 retries
        escalates with trigger ``persistent_test_failure``.
        """
        if self._trigger_fired:
            return None

        self._test_failure_count += 1
        logger.debug(
            "Test failure recorded: %d/%d",
            self._test_failure_count, _PERSISTENT_TEST_FAILURE_THRESHOLD,
        )

        if self._test_failure_count >= _PERSISTENT_TEST_FAILURE_THRESHOLD:
            self._trigger_fired = True
            logger.warning(
                "Uncertainty escalation: persistent_test_failure (count=%d)",
                self._test_failure_count,
            )
            return UncertaintyTrigger(
                trigger="persistent_test_failure",
                agent_name=agent_name,
                step_index=step_index,
                detail={
                    "consecutive_failures": self._test_failure_count,
                    "threshold": _PERSISTENT_TEST_FAILURE_THRESHOLD,
                },
            )
        return None

    def record_test_success(self) -> None:
        """Record a successful test run, resetting the counter."""
        self._test_failure_count = 0

    # ── Path B: Same diff_hash rejected twice ────────────────────

    def record_diff_rejection(
        self,
        diff_hash: str,
        agent_name: str,
        step_index: int,
    ) -> UncertaintyTrigger | None:
        """Record a Reviewer rejection of a diff_hash.

        Returns an UncertaintyTrigger if the same diff_hash has been
        rejected 2 times, or None otherwise.

        VAL-UNCERTAINTY-003: Reviewer rejects same diff_hash twice
        escalates with trigger ``same_fix_rejected_twice``.
        """
        if self._trigger_fired:
            return None

        count = self._rejected_diff_hashes.get(diff_hash, 0) + 1
        self._rejected_diff_hashes[diff_hash] = count
        logger.debug(
            "Diff rejection recorded: hash=%s… → count=%d",
            diff_hash[:12], count,
        )

        if count >= 2:
            self._trigger_fired = True
            logger.warning(
                "Uncertainty escalation: same_fix_rejected_twice (hash=%s…)",
                diff_hash[:12],
            )
            return UncertaintyTrigger(
                trigger="same_fix_rejected_twice",
                agent_name=agent_name,
                step_index=step_index,
                detail={
                    "diff_hash": diff_hash,
                    "rejection_count": count,
                },
            )
        return None

    # ── Path B: Tool-error rate tracking ─────────────────────────

    def record_tool_call(
        self,
        agent_name: str,
        success: bool,
        step_index: int = 0,
    ) -> UncertaintyTrigger | None:
        """Record a tool call outcome (success or error).

        Returns an UncertaintyTrigger if the error rate in the last
        10 calls exceeds 50%, or None otherwise.

        VAL-UNCERTAINTY-004: Tool-error rate >50% in 10-call window
        escalates with trigger ``tool_error_rate_exceeded``.
        """
        if self._trigger_fired:
            return None

        if agent_name not in self._tool_error_windows:
            self._tool_error_windows[agent_name] = deque(
                maxlen=_TOOL_ERROR_WINDOW_SIZE
            )

        window = self._tool_error_windows[agent_name]
        window.append(success)

        # Only check rate when we have at least 1 call in the window
        total = len(window)
        errors = sum(1 for s in window if not s)
        error_rate = errors / total if total > 0 else 0.0

        logger.debug(
            "Tool call recorded: agent=%s, success=%s, errors=%d/%d (%.0f%%)",
            agent_name, success, errors, total, error_rate * 100,
        )

        if total >= 2 and error_rate > _TOOL_ERROR_RATE_THRESHOLD:
            self._trigger_fired = True
            logger.warning(
                "Uncertainty escalation: tool_error_rate_exceeded "
                "for agent=%s (%d errors in %d calls)",
                agent_name, errors, total,
            )
            return UncertaintyTrigger(
                trigger="tool_error_rate_exceeded",
                agent_name=agent_name,
                step_index=step_index,
                detail={
                    "errors": errors,
                    "total": total,
                    "rate": round(error_rate, 4),
                    "window": _TOOL_ERROR_WINDOW_SIZE,
                },
            )
        return None

    # ── General ──────────────────────────────────────────────────

    @property
    def has_fired(self) -> bool:
        """Whether an uncertainty trigger has already fired for this task."""
        return self._trigger_fired

    def reset(self) -> None:
        """Reset all tracking state (for new task)."""
        self._pydantic_fail_counters.clear()
        self._test_failure_count = 0
        self._rejected_diff_hashes.clear()
        self._tool_error_windows.clear()
        self._trigger_fired = False
