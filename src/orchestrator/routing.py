"""Conditional-edge routing functions for the supervisor_only/hybrid graphs.

These synchronous, pure decision functions inspect the
``OrchestratorState`` after each agent runs (planner, coder, review, qa,
commit) and return the name of the next node to execute.  They encode
the supervisor's routing logic: guardrail/loop/uncertainty escalations
go to HITL nodes, cost-budget or failure outcomes halt, and otherwise
the graph advances sequentially (planner → coder → reviewer → qa →
finalize).  They are re-exported from ``supervisor_only`` so existing
imports and test patches that target that path keep working.
"""

from __future__ import annotations

import logging
from typing import Literal

from src.orchestrator import OrchestratorState

logger = logging.getLogger(__name__)


def route_after_planner(
    state: OrchestratorState,
) -> Literal[
    "halt_cost_budget_exhausted",
    "halt_task_failed",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "run_coder",
]:
    """After Planner, route to guardrail escalation, loop detection,
    uncertainty escalation, or Coder.

    VAL-GUARDRAIL-009: If a guardrail violation was detected, route
    to the HITL escalation node.
    VAL-LOOP-DETECT-004: If loop was detected, route to HITL.
    VAL-UNCERTAINTY-007: If uncertainty escalation, route to HITL.
    """
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
    return "run_coder"


def route_after_coder(
    state: OrchestratorState,
) -> Literal[
    "halt_cost_budget_exhausted",
    "halt_task_failed",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "run_reviewer",
]:
    """After Coder, route to guardrail escalation, loop detection,
    uncertainty escalation, or Reviewer.
    """
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


def route_after_review(
    state: OrchestratorState,
) -> Literal[
    "halt_cost_budget_exhausted",
    "halt_task_failed",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "run_qa",
    "run_coder",
    "run_supervisor_finalize",
    "halt_retry_exhausted",
]:
    """After Reviewer, route based on verdict.

    - ``guardrail_block``: escalate to HITL (VAL-GUARDRAIL-009)
    - ``loop_detected``: escalate to HITL (VAL-LOOP-DETECT-004)
    - ``uncertainty_escalation``: escalate to HITL (VAL-UNCERTAINTY-007)
    - ``accept``: advance to QA (VAL-QA-001, VAL-QA-003)
    - ``reject_with_changes``: re-route to Coder (if retry budget
      allows), otherwise halt with retry_budget_exhausted
    - ``reject``: advance to Supervisor finalize (terminal rejection)

    VAL-TOPOLOGY-003: In supervisor_only, the re-route goes through
    the Supervisor (this routing function IS the Supervisor's
    routing logic).  There is NO direct Reviewer→Coder peer edge.

    VAL-REVIEWER-004: When ReviewResult.verdict == 'accept', the
    orchestrator routes to QA next, not back to Coder.
    """
    from src.orchestrator.supervisor_only import _MAX_RETRIES_PER_STEP

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
        # No review result — should not happen, advance to finalize
        logger.warning("route_after_review called with no review_result")
        return "run_supervisor_finalize"

    if review.verdict == "accept":
        # Accept: advance to QA (VAL-REVIEWER-004, VAL-QA-001)
        return "run_qa"

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


def route_after_qa(
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
    """After QA, route based on test results.

    - ``guardrail_block``: escalate to HITL (VAL-GUARDRAIL-009)
    - ``loop_detected``: escalate to HITL (VAL-LOOP-DETECT-004)
    - ``uncertainty_escalation``: escalate to HITL (VAL-UNCERTAINTY-007)
    - All pass (failed == 0): advance to Supervisor finalize (→ PR)
    - Failures (failed > 0): halt with test failure (VAL-QA-005)

    VAL-QA-005: When TestReport.failed > 0, the orchestrator does
    NOT proceed to PR creation. It either retries (within budget)
    or escalates to HITL with cause persistent_test_failure.
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

    report = state.test_report
    if report is None:
        # No test report — should not happen, advance to finalize
        logger.warning("route_after_qa called with no test_report")
        return "run_supervisor_finalize"

    if report.failed > 0:
        # Test failures: do NOT open PR (VAL-QA-005)
        logger.warning(
            "QA found %d failing test(s) for task %s — NOT opening PR",
            report.failed,
            state.task_id,
        )
        return "halt_test_failure"

    # All tests pass: advance to finalize
    return "run_supervisor_finalize"


def route_after_commit_and_push(
    state: OrchestratorState,
) -> Literal["run_open_pr", "halt_github_delivery_failed"]:
    """Only continue to PR creation after a successful commit/push."""
    if state.outcome == "github_delivery_failed" or state.status == "failed":
        return "halt_github_delivery_failed"
    return "run_open_pr"
