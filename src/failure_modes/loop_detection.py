"""Loop detection — sliding window of tool calls per agent.

Architecture §2.9: Maintain a sliding window of the last 5 tool
calls (per agent).  If the same ``(tool_name, sha256(canonical_json_args))``
appears 3+ times in the window, halt that agent and escalate
with ``outcome='loop_detected'``.

Canonicalization: sorted keys, no whitespace (VAL-LOOP-DETECT-002).
Window size: 5 (VAL-LOOP-DETECT-003).
Threshold: 3 identical calls in the window (VAL-LOOP-DETECT-001).
Per-agent: each agent has an independent sliding window (VAL-LOOP-DETECT-005).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# Default window size — the last N tool calls to examine
DEFAULT_WINDOW_SIZE: int = 5

# Default threshold — how many identical calls trigger a halt
DEFAULT_THRESHOLD: int = 3


def canonical_json(args: dict[str, Any]) -> str:
    """Canonicalize a JSON args dict: sorted keys, no whitespace.

    VAL-LOOP-DETECT-002: Two semantically identical arg dicts with
    different key ordering or whitespace produce the same canonical
    string.  E.g., ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    both produce ``'{"a":1,"b":2}'``.
    """
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def compute_args_hash(args: dict[str, Any]) -> str:
    """Compute SHA-256 hash of the canonical JSON of args.

    Two semantically identical args produce the same hash.
    Two semantically distinct args produce different hashes.
    """
    canonical = canonical_json(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolCallRecord:
    """A single recorded tool call for loop detection.

    Stores the tool name and the SHA-256 hash of the canonical
    JSON of the arguments.
    """

    __slots__ = ("tool_name", "args_hash")

    def __init__(self, tool_name: str, args_hash: str) -> None:
        self.tool_name = tool_name
        self.args_hash = args_hash

    @property
    def key(self) -> str:
        """Composite key for matching: ``tool_name:args_hash``."""
        return f"{self.tool_name}:{self.args_hash}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCallRecord):
            return NotImplemented
        return self.tool_name == other.tool_name and self.args_hash == other.args_hash

    def __hash__(self) -> int:
        return hash((self.tool_name, self.args_hash))


class LoopDetector:
    """Per-agent sliding-window loop detector.

    Each agent maintains an independent sliding window
    (VAL-LOOP-DETECT-005).  When the same ``(tool_name, args_hash)``
    appears ``threshold`` times within the last ``window_size``
    calls, ``check()`` returns a ``LoopDetected`` result.

    Usage::

        detector = LoopDetector(window_size=5, threshold=3)
        for _ in range(3):
            detector.record("coder", "apply_diff", {"file": "foo.py"})
        result = detector.check("coder")
        assert result is not None  # loop_detected!
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> None:
        if window_size < 1:
            msg = f"window_size must be >= 1, got {window_size}"
            raise ValueError(msg)
        if threshold < 2:
            msg = f"threshold must be >= 2, got {threshold}"
            raise ValueError(msg)
        self._window_size = window_size
        self._threshold = threshold
        self._windows: dict[str, deque[ToolCallRecord]] = {}

    @property
    def window_size(self) -> int:
        """Return the configured window size."""
        return self._window_size

    @property
    def threshold(self) -> int:
        """Return the configured threshold."""
        return self._threshold

    def record(
        self,
        agent_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolCallRecord:
        """Record a tool call for an agent.

        Adds the call to the agent's sliding window, trimming
        to ``window_size``.  Returns the ``ToolCallRecord`` for
        inspection.
        """
        args_hash = compute_args_hash(args)
        record = ToolCallRecord(tool_name=tool_name, args_hash=args_hash)

        if agent_name not in self._windows:
            self._windows[agent_name] = deque(maxlen=self._window_size)
        self._windows[agent_name].append(record)

        logger.debug(
            "Loop detector record: agent=%s, tool=%s, hash=%s…",
            agent_name, tool_name, args_hash[:12],
        )
        return record

    def get_window(self, agent_name: str) -> list[ToolCallRecord]:
        """Return the current sliding window for an agent."""
        if agent_name not in self._windows:
            return []
        return list(self._windows[agent_name])

    def check(self, agent_name: str) -> LoopDetectionResult | None:
        """Check if a loop is detected for the given agent.

        Returns a ``LoopDetectionResult`` if the same
        ``(tool_name, args_hash)`` appears ``threshold`` times
        within the last ``window_size`` calls, or ``None`` if
        no loop is detected.
        """
        window = self.get_window(agent_name)
        if len(window) < self._threshold:
            return None

        # Count occurrences of each key in the window
        counts: dict[str, int] = {}
        for rec in window:
            counts[rec.key] = counts.get(rec.key, 0) + 1

        # Find any key that meets the threshold
        for key, count in counts.items():
            if count >= self._threshold:
                # Extract tool_name and args_hash from the key
                tool_name, args_hash = key.split(":", 1)
                logger.warning(
                    "Loop detected for agent=%s: tool=%s, count=%d, threshold=%d",
                    agent_name, tool_name, count, self._threshold,
                )
                return LoopDetectionResult(
                    agent_name=agent_name,
                    tool_name=tool_name,
                    args_hash=args_hash,
                    count=count,
                    window_size=self._window_size,
                    threshold=self._threshold,
                    window_snapshot=[
                        {"tool_name": r.tool_name, "args_hash": r.args_hash}
                        for r in window
                    ],
                )
        return None

    def reset(self, agent_name: str) -> None:
        """Clear the sliding window for a specific agent."""
        self._windows.pop(agent_name, None)


class LoopDetectionResult:
    """Result returned when a loop is detected.

    Contains all the context needed for Langfuse logging,
    outcomes row creation, and HITL escalation.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        tool_name: str,
        args_hash: str,
        count: int,
        window_size: int,
        threshold: int,
        window_snapshot: list[dict[str, str]],
    ) -> None:
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.args_hash = args_hash
        self.count = count
        self.window_size = window_size
        self.threshold = threshold
        self.window_snapshot = window_snapshot

    def to_outcome_data(self) -> dict[str, Any]:
        """Return a dict for writing an ``outcomes`` row."""
        return {
            "outcome": "loop_detected",
            "detail": {
                "agent_name": self.agent_name,
                "tool_name": self.tool_name,
                "args_hash": self.args_hash,
                "count": self.count,
                "window_size": self.window_size,
                "threshold": self.threshold,
                "trigger": "loop_detected",
                "window_snapshot": self.window_snapshot,
            },
        }

    def to_hitl_details(self) -> dict[str, Any]:
        """Return HITL escalation details for the interrupt payload."""
        return {
            "cause": "loop_detected",
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "count": self.count,
            "explanation": (
                f"Agent '{self.agent_name}' called tool '{self.tool_name}' "
                f"with identical arguments {self.count} times within the last "
                f"{self.window_size} calls (threshold={self.threshold})."
            ),
            "window_snapshot": self.window_snapshot,
        }
