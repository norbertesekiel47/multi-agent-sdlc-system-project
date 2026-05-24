"""Retry budget — per (agent, step) counter with max 3 attempts.

Architecture §2.9: After 3 attempts at the same step (Pydantic
validation failures, tool errors, etc.), halt that step and
escalate with ``outcome='retry_budget_exhausted'``.

Each ``(agent_name, step_index)`` tuple has an independent counter.
Failures on one agent/step do NOT consume another's budget
(VAL-RETRY-003).

The default max retries is 3 (VAL-RETRY-001) and is configurable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default maximum retries per (agent, step) — VAL-RETRY-001
DEFAULT_MAX_RETRIES: int = 3


class RetryBudget:
    """Tracks per-(agent, step) retry counters.

    Each call to ``increment()`` records an attempt for the given
    ``(agent_name, step_index)`` pair.  When the count reaches
    ``max_retries``, ``is_exhausted()`` returns ``True`` and
    the orchestrator must halt + escalate.

    Usage::

        budget = RetryBudget(max_retries=3)
        budget.increment("coder", 0)  # attempt 1
        budget.increment("coder", 0)  # attempt 2
        budget.increment("coder", 0)  # attempt 3
        assert budget.is_exhausted("coder", 0)  # True — halt!
    """

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        if max_retries < 1:
            msg = f"max_retries must be >= 1, got {max_retries}"
            raise ValueError(msg)
        self._max_retries = max_retries
        self._counters: dict[str, int] = {}

    @property
    def max_retries(self) -> int:
        """Return the configured max retries per (agent, step)."""
        return self._max_retries

    def _key(self, agent_name: str, step_index: int) -> str:
        """Build the counter key from agent name and step index."""
        return f"{agent_name}:{step_index}"

    def increment(self, agent_name: str, step_index: int) -> int:
        """Record an attempt for (agent, step).

        Returns the new count after incrementing.
        """
        key = self._key(agent_name, step_index)
        current = self._counters.get(key, 0)
        new_count = current + 1
        self._counters[key] = new_count
        logger.debug(
            "Retry budget increment: %s → %d/%d",
            key, new_count, self._max_retries,
        )
        return new_count

    def get_count(self, agent_name: str, step_index: int) -> int:
        """Return the current retry count for (agent, step)."""
        key = self._key(agent_name, step_index)
        return self._counters.get(key, 0)

    def is_exhausted(self, agent_name: str, step_index: int) -> bool:
        """Check if the retry budget is exhausted for (agent, step).

        VAL-RETRY-002: Returns True when the count equals or exceeds
        max_retries — no fourth attempt should be made.
        """
        return self.get_count(agent_name, step_index) >= self._max_retries

    def reset(self, agent_name: str, step_index: int) -> None:
        """Reset the retry counter for a specific (agent, step)."""
        key = self._key(agent_name, step_index)
        self._counters.pop(key, None)

    def to_state(self) -> dict[str, int]:
        """Serialize the counters for storage in OrchestratorState."""
        return dict(self._counters)

    @classmethod
    def from_state(
        cls,
        data: dict[str, int],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> RetryBudget:
        """Restore a RetryBudget from serialized state."""
        budget = cls(max_retries=max_retries)
        budget._counters = dict(data)
        return budget
