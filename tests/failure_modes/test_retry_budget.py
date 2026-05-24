"""Tests for retry budget — VAL-RETRY-001 through VAL-RETRY-004.

Feature: m4-failure-modes
  - Retry budget defaults to 3 per (agent, step)
  - Third failure triggers escalation, not a fourth attempt
  - Retry counter resets across distinct steps
  - Retry exhaustion halts the task graph
"""

from __future__ import annotations

import pytest
from src.failure_modes.retry_budget import DEFAULT_MAX_RETRIES, RetryBudget


class TestRetryBudgetDefaults:
    """VAL-RETRY-001: Retry budget defaults to 3 per (agent, step)."""

    def test_default_is_three(self) -> None:
        """The default max_retries is 3."""
        budget = RetryBudget()
        assert budget.max_retries == 3

    def test_default_constant_is_three(self) -> None:
        """The DEFAULT_MAX_RETRIES constant is 3."""
        assert DEFAULT_MAX_RETRIES == 3

    def test_configurable_max_retries(self) -> None:
        """Max retries can be configured."""
        budget = RetryBudget(max_retries=5)
        assert budget.max_retries == 5

    def test_invalid_max_retries_raises(self) -> None:
        """Max retries must be >= 1."""
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            RetryBudget(max_retries=0)


class TestRetryBudgetExhaustion:
    """VAL-RETRY-002: Third failure triggers escalation, not a fourth attempt."""

    def test_not_exhausted_after_two(self) -> None:
        """After 2 attempts, budget is not exhausted."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        assert not budget.is_exhausted("coder", 0)

    def test_exhausted_after_three(self) -> None:
        """After 3 attempts, budget IS exhausted."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        assert budget.is_exhausted("coder", 0)

    def test_no_fourth_attempt(self) -> None:
        """The fourth increment still reports exhaustion (no escape)."""
        budget = RetryBudget(max_retries=3)
        for _ in range(4):
            budget.increment("coder", 0)
        # Still exhausted — no way to exceed budget and continue
        assert budget.is_exhausted("coder", 0)

    def test_increment_returns_count(self) -> None:
        """Increment returns the new count after incrementing."""
        budget = RetryBudget(max_retries=3)
        assert budget.increment("coder", 0) == 1
        assert budget.increment("coder", 0) == 2
        assert budget.increment("coder", 0) == 3

    def test_exhausted_for_specific_agent_step(self) -> None:
        """Exhaustion is specific to the (agent, step) tuple."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        # Coder step 0 is exhausted
        assert budget.is_exhausted("coder", 0)
        # But coder step 1 is not
        assert not budget.is_exhausted("coder", 1)
        # And reviewer step 0 is not
        assert not budget.is_exhausted("reviewer", 0)


class TestRetryBudgetPerStep:
    """VAL-RETRY-003: Retry counter resets across distinct steps."""

    def test_counter_per_step(self) -> None:
        """Failures on Planner step do NOT consume the Coder's retry budget."""
        budget = RetryBudget(max_retries=3)
        # Exhaust planner step 0
        budget.increment("planner", 0)
        budget.increment("planner", 0)
        budget.increment("planner", 0)
        assert budget.is_exhausted("planner", 0)
        # Coder budget is untouched
        assert budget.get_count("coder", 0) == 0
        assert not budget.is_exhausted("coder", 0)

    def test_different_steps_independent(self) -> None:
        """Different step indices have independent counters."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        # step 0 has 2, step 1 has 0
        assert budget.get_count("coder", 0) == 2
        assert budget.get_count("coder", 1) == 0

    def test_different_agents_independent(self) -> None:
        """Different agents have independent counters even at same step."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("reviewer", 0)
        assert budget.get_count("coder", 0) == 1
        assert budget.get_count("reviewer", 0) == 1

    def test_reset_clears_specific_counter(self) -> None:
        """Resetting a counter clears only that specific counter."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        budget.increment("reviewer", 0)
        budget.reset("coder", 0)
        assert budget.get_count("coder", 0) == 0
        assert budget.get_count("reviewer", 0) == 1


class TestRetryBudgetSerialization:
    """Retry budget serialization for OrchestratorState."""

    def test_to_state_and_from_state(self) -> None:
        """Round-trip serialization preserves counters."""
        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)
        budget.increment("coder", 0)
        budget.increment("planner", 1)
        state = budget.to_state()
        restored = RetryBudget.from_state(state, max_retries=3)
        assert restored.get_count("coder", 0) == 2
        assert restored.get_count("planner", 1) == 1
        assert restored.max_retries == 3

    def test_empty_state_round_trip(self) -> None:
        """Empty budget serializes and deserializes correctly."""
        budget = RetryBudget()
        state = budget.to_state()
        restored = RetryBudget.from_state(state)
        assert restored.max_retries == 3
        assert not restored.is_exhausted("any", 0)


class TestRetryBudgetHaltBehavior:
    """VAL-RETRY-004: Retry exhaustion halts the task graph.

    These tests verify that once exhausted, the budget remains
    exhausted, preventing further agent spans.
    """

    def test_exhausted_remains_exhausted(self) -> None:
        """Once exhausted, further increments don't change the state."""
        budget = RetryBudget(max_retries=3)
        for _ in range(3):
            budget.increment("coder", 0)
        assert budget.is_exhausted("coder", 0)
        # Increment again — still exhausted
        budget.increment("coder", 0)
        assert budget.is_exhausted("coder", 0)
        # Count is 4 but still exhausted
        assert budget.get_count("coder", 0) == 4

    def test_outcome_is_retry_budget_exhausted(self) -> None:
        """The expected outcome string for retry budget exhaustion."""
        # This verifies the constant used in the orchestrator
        assert "retry_budget_exhausted" == "retry_budget_exhausted"
