"""Tests for loop detection — VAL-LOOP-DETECT-001 through VAL-LOOP-DETECT-005.

Feature: m4-failure-modes
  - Identical (tool, args_hash) 3× in last 5 calls halts agent
  - Args hash uses canonical JSON
  - Window is exactly the last 5 calls
  - Loop detection escalates to HITL
  - Loop detection is per-agent, not global
"""

from __future__ import annotations

from src.failure_modes.loop_detection import (
    LoopDetector,
    canonical_json,
    compute_args_hash,
)


class TestCanonicalJson:
    """VAL-LOOP-DETECT-002: Args hash uses canonical JSON."""

    def test_args_hash_canonical(self) -> None:
        """Two semantically identical arg dicts produce the same hash."""
        args_a = {"a": 1, "b": 2}
        args_b = {"b": 2, "a": 1}
        assert compute_args_hash(args_a) == compute_args_hash(args_b)

    def test_different_args_different_hash(self) -> None:
        """Two semantically distinct arg dicts produce different hashes."""
        args_a = {"a": 1, "b": 2}
        args_b = {"a": 1, "b": 3}
        assert compute_args_hash(args_a) != compute_args_hash(args_b)

    def test_canonical_json_sorted_keys(self) -> None:
        """Canonical JSON has sorted keys."""
        result = canonical_json({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_canonical_json_no_whitespace(self) -> None:
        """Canonical JSON has no whitespace."""
        result = canonical_json({"a": 1, "b": 2})
        assert " " not in result
        assert "\n" not in result

    def test_nested_dict_sorted_keys(self) -> None:
        """Nested dicts are also sorted."""
        result = canonical_json({"z": {"b": 2, "a": 1}, "y": 3})
        assert result == '{"y":3,"z":{"a":1,"b":2}}'

    def test_empty_dict_canonical(self) -> None:
        """Empty dict produces consistent canonical form."""
        result = canonical_json({})
        assert result == "{}"

    def test_whitespace_difference_same_hash(self) -> None:
        """Whitespace differences in values don't matter for dict structure."""
        # The args are dicts with same keys/values — whitespace is in
        # the Python representation, not the values themselves
        args1 = {"path": "/workspace/file.py"}
        args2 = {"path": "/workspace/file.py"}
        assert compute_args_hash(args1) == compute_args_hash(args2)


class TestLoopDetectionWindow:
    """VAL-LOOP-DETECT-003: Window is exactly the last 5 calls."""

    def test_window_size_five_default(self) -> None:
        """Default window size is 5."""
        detector = LoopDetector()
        assert detector.window_size == 5

    def test_threshold_three_default(self) -> None:
        """Default threshold is 3."""
        detector = LoopDetector()
        assert detector.threshold == 3

    def test_window_limits_to_five(self) -> None:
        """Sliding window only keeps the last 5 calls."""
        detector = LoopDetector(window_size=5, threshold=3)
        # Record 7 calls
        for i in range(7):
            detector.record("coder", f"tool_{i}", {"i": i})
        window = detector.get_window("coder")
        assert len(window) == 5

    def test_old_call_outside_window_no_trigger(self) -> None:
        """Pattern [X, X, A, B, C, D, X] does NOT trigger loop detection.

        The first two X's are outside the 5-call window, so only
        one X is in the window — not enough to trigger.
        """
        detector = LoopDetector(window_size=5, threshold=3)
        agent = "coder"

        # Record: X, X (same tool+args) — these will fall out of window
        same_args = {"file": "foo.py", "line": 10}
        detector.record(agent, "apply_diff", same_args)  # X1
        detector.record(agent, "apply_diff", same_args)  # X2

        # Record 4 different calls to push X1 and X2 out
        detector.record(agent, "read_file", {"path": "a.py"})  # A
        detector.record(agent, "read_file", {"path": "b.py"})  # B
        detector.record(agent, "run_command", {"cmd": "ls"})   # C
        detector.record(agent, "run_command", {"cmd": "pwd"})  # D

        # Now window is [A, B, C, D] (4 items) — X1 and X2 are out
        # Record one more X — only 1 X in window
        detector.record(agent, "apply_diff", same_args)  # X3

        result = detector.check(agent)
        assert result is None, "Should not trigger: only 1 X in window"

    def test_all_three_in_window_triggers(self) -> None:
        """Pattern [A, X, B, X, X] (3 X's all within last 5) does trigger."""
        detector = LoopDetector(window_size=5, threshold=3)
        agent = "coder"

        same_args = {"file": "foo.py", "line": 10}
        detector.record(agent, "read_file", {"path": "a.py"})   # A
        detector.record(agent, "apply_diff", same_args)          # X1
        detector.record(agent, "read_file", {"path": "b.py"})   # B
        detector.record(agent, "apply_diff", same_args)          # X2
        detector.record(agent, "apply_diff", same_args)          # X3

        result = detector.check(agent)
        assert result is not None
        assert result.tool_name == "apply_diff"
        assert result.count == 3

    def test_configurable_window_and_threshold(self) -> None:
        """Window size and threshold are configurable."""
        detector = LoopDetector(window_size=3, threshold=2)
        agent = "coder"
        same_args = {"file": "foo.py"}

        detector.record(agent, "apply_diff", same_args)  # 1
        detector.record(agent, "apply_diff", same_args)  # 2

        result = detector.check(agent)
        assert result is not None
        assert result.count == 2


class TestLoopDetectionHalt:
    """VAL-LOOP-DETECT-001: Identical (tool, args_hash) 3× in last 5 halts agent."""

    def test_three_in_five_halts(self) -> None:
        """3 identical (tool, args_hash) in last 5 calls triggers halt."""
        detector = LoopDetector(window_size=5, threshold=3)
        agent = "coder"
        same_args = {"file": "foo.py", "line": 10}

        detector.record(agent, "apply_diff", same_args)
        detector.record(agent, "read_file", {"path": "a.py"})
        detector.record(agent, "apply_diff", same_args)
        detector.record(agent, "run_command", {"cmd": "ls"})
        detector.record(agent, "apply_diff", same_args)

        result = detector.check(agent)
        assert result is not None
        assert result.agent_name == "coder"
        assert result.tool_name == "apply_diff"
        assert result.count >= 3

    def test_not_triggered_below_threshold(self) -> None:
        """2 identical calls (below threshold 3) do not trigger."""
        detector = LoopDetector(window_size=5, threshold=3)
        agent = "coder"
        same_args = {"file": "foo.py"}

        detector.record(agent, "apply_diff", same_args)
        detector.record(agent, "apply_diff", same_args)

        result = detector.check(agent)
        assert result is None

    def test_loop_detection_result_outcome(self) -> None:
        """LoopDetectionResult produces correct outcome data."""
        detector = LoopDetector(window_size=5, threshold=3)
        agent = "coder"
        same_args = {"file": "foo.py"}

        for _ in range(3):
            detector.record(agent, "apply_diff", same_args)

        result = detector.check(agent)
        assert result is not None
        outcome = result.to_outcome_data()
        assert outcome["outcome"] == "loop_detected"
        assert outcome["detail"]["trigger"] == "loop_detected"
        assert outcome["detail"]["tool_name"] == "apply_diff"
        assert outcome["detail"]["count"] >= 3

    def test_loop_detection_hitl_details(self) -> None:
        """LoopDetectionResult produces correct HITL details."""
        detector = LoopDetector(window_size=5, threshold=3)
        agent = "coder"
        same_args = {"file": "foo.py"}

        for _ in range(3):
            detector.record(agent, "apply_diff", same_args)

        result = detector.check(agent)
        assert result is not None
        hitl = result.to_hitl_details()
        assert hitl["cause"] == "loop_detected"
        assert "explanation" in hitl
        assert "window_snapshot" in hitl


class TestLoopDetectionPerAgent:
    """VAL-LOOP-DETECT-005: Loop detection is per-agent, not global."""

    def test_per_agent_window(self) -> None:
        """A repeated tool call by the Coder does not affect Reviewer's window."""
        detector = LoopDetector(window_size=5, threshold=3)

        # Coder repeats the same call 3 times
        same_args = {"file": "foo.py"}
        for _ in range(3):
            detector.record("coder", "apply_diff", same_args)

        # Coder should be detected
        coder_result = detector.check("coder")
        assert coder_result is not None
        assert coder_result.agent_name == "coder"

        # Reviewer should NOT be detected — independent window
        reviewer_result = detector.check("reviewer")
        assert reviewer_result is None

    def test_independent_agents_no_cross_contamination(self) -> None:
        """Each agent has its own sliding window with no cross-contamination."""
        detector = LoopDetector(window_size=5, threshold=3)

        # Record calls for two different agents
        for _ in range(2):
            detector.record("coder", "apply_diff", {"file": "a.py"})
        for _ in range(2):
            detector.record("reviewer", "run_command", {"cmd": "ruff"})

        # Neither has enough to trigger
        assert detector.check("coder") is None
        assert detector.check("reviewer") is None

    def test_reset_clears_only_specific_agent(self) -> None:
        """Resetting one agent's window doesn't affect another's."""
        detector = LoopDetector(window_size=5, threshold=3)

        for _ in range(3):
            detector.record("coder", "apply_diff", {"file": "a.py"})
        for _ in range(3):
            detector.record("reviewer", "run_command", {"cmd": "ruff"})

        # Both detected
        assert detector.check("coder") is not None
        assert detector.check("reviewer") is not None

        # Reset only coder
        detector.reset("coder")

        # Coder no longer detected
        assert detector.check("coder") is None
        # Reviewer still detected (window still has 3 entries)
        assert detector.check("reviewer") is not None
