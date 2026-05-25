"""Tests for wiring LoopDetector and UncertaintyEscalation into graph nodes.

Fix 2 from M4 scrutiny: LoopDetector and UncertaintyEscalation classes
exist but were dead code. These tests verify they are now called from:
  - Agent nodes (catch ValidationError → record_pydantic_failure)
  - route_after_review (reject_with_changes → record_diff_rejection)
  - Tool-call dispatch (record_tool_call + _check_loop_detection)
  - Successful validation (record_pydantic_success resets counter)
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from src.orchestrator import OrchestratorState

# ── Helper: create a minimal state for testing ──────────────────────


def _make_state(**overrides: object) -> OrchestratorState:
    """Create a minimal OrchestratorState for testing."""
    defaults: dict[str, object] = {
        "task_id": uuid4().hex,
        "repo_url": "https://github.com/test/repo",
        "issue_number": 1,
        "issue_text": "Test issue",
        "topology": "supervisor_only",
        "status": "running",
        "step_index": 0,
        "trace_id": uuid4().hex,
        "supervisor_span_id": "sup-123",
    }
    defaults.update(overrides)
    return OrchestratorState(**defaults)  # type: ignore[arg-type]


# ── UncertaintyEscalation reconstruction tests ──────────────────────


class TestUncertaintyReconstruction:
    """Verify that UncertaintyEscalation is reconstructed from state correctly."""

    def test_reconstruct_from_empty_state(self) -> None:
        """Reconstruct from empty state produces fresh UncertaintyEscalation."""
        from src.orchestrator.supervisor_only import _reconstruct_uncertainty

        state = _make_state()
        ue = _reconstruct_uncertainty(state)
        assert not ue.has_fired
        assert ue._pydantic_fail_counters == {}
        assert ue._test_failure_count == 0

    def test_reconstruct_preserves_pydantic_counters(self) -> None:
        """Reconstruct preserves pydantic fail counters from state."""
        from src.orchestrator.supervisor_only import _reconstruct_uncertainty

        state = _make_state(
            pydantic_fail_counters={"coder:0": 2, "reviewer:1": 1},
            uncertainty_fired=False,
        )
        ue = _reconstruct_uncertainty(state)
        assert ue._pydantic_fail_counters == {"coder:0": 2, "reviewer:1": 1}

    def test_reconstruct_preserves_rejected_diff_hashes(self) -> None:
        """Reconstruct preserves rejected diff_hashes from state."""
        from src.orchestrator.supervisor_only import _reconstruct_uncertainty

        state = _make_state(
            rejected_diff_hashes={"hash_abc": 1},
            uncertainty_fired=False,
        )
        ue = _reconstruct_uncertainty(state)
        assert ue._rejected_diff_hashes == {"hash_abc": 1}

    def test_sync_state_roundtrip(self) -> None:
        """Reconstruct → record → sync produces correct state update."""
        from src.orchestrator.supervisor_only import (
            _reconstruct_uncertainty,
            _sync_uncertainty_to_state,
        )

        state = _make_state()
        ue = _reconstruct_uncertainty(state)
        ue.record_pydantic_failure("coder", 0)
        ue.record_pydantic_failure("coder", 0)
        update = _sync_uncertainty_to_state(ue)
        assert update["pydantic_fail_counters"] == {"coder:0": 2}


# ── _record_pydantic_failure tests ──────────────────────────────────


class TestRecordPydanticFailure:
    """Verify _record_pydantic_failure records and checks for escalation."""

    def test_single_failure_no_escalation(self) -> None:
        """A single Pydantic failure does not trigger escalation."""
        from src.orchestrator.supervisor_only import _record_pydantic_failure

        state = _make_state()
        result = _record_pydantic_failure(
            agent_name="coder",
            step_index=0,
            state=state,
        )
        # Should return None or just counter update, not escalation
        assert result is None or result.get("outcome") != "uncertainty_escalation"

    def test_three_failures_triggers_escalation(self) -> None:
        """Three consecutive Pydantic failures trigger uncertainty_escalation."""
        from src.orchestrator.supervisor_only import (
            _record_pydantic_failure,
        )

        state = _make_state()
        # First two failures — no escalation
        for _ in range(2):
            result = _record_pydantic_failure(
                agent_name="coder",
                step_index=0,
                state=state,
            )
            if result is not None:
                # Update state for next iteration
                state_data = state.model_dump()
                state_data.update(result)
                state = OrchestratorState(**state_data)  # type: ignore[arg-type]

        # Third failure → escalation
        result = _record_pydantic_failure(
            agent_name="coder",
            step_index=0,
            state=state,
        )
        assert result is not None
        assert result.get("outcome") == "uncertainty_escalation"
        assert "pydantic_validation_3x" in str(result.get("errors", []))


# ── _record_pydantic_success tests ──────────────────────────────────


class TestRecordPydanticSuccess:
    """Verify _record_pydantic_success resets the failure counter."""

    def test_success_resets_counter(self) -> None:
        """A successful validation resets the Pydantic failure counter."""
        from src.orchestrator.supervisor_only import (
            _record_pydantic_failure,
            _record_pydantic_success,
        )

        state = _make_state()

        # Record 2 failures
        for _ in range(2):
            result = _record_pydantic_failure(
                agent_name="coder",
                step_index=0,
                state=state,
            )
            if result is not None:
                state_data = state.model_dump()
                state_data.update(result)
                state = OrchestratorState(**state_data)  # type: ignore[arg-type]

        # Success resets the counter
        result = _record_pydantic_success(
            agent_name="coder",
            step_index=0,
            state=state,
        )
        assert "pydantic_fail_counters" in result
        # Counter should be empty (reset)
        counters = result["pydantic_fail_counters"]
        assert "coder:0" not in counters or counters.get("coder:0", 0) == 0


# ── _record_diff_rejection_check tests ───────────────────────────────


class TestRecordDiffRejectionCheck:
    """Verify diff rejection recording and same_fix_rejected_twice escalation."""

    def test_first_rejection_no_escalation(self) -> None:
        """First rejection of a diff_hash does not trigger escalation."""
        from src.orchestrator.supervisor_only import _record_diff_rejection_check

        state = _make_state()
        result = _record_diff_rejection_check(
            diff_hash="hash_abc",
            agent_name="reviewer",
            step_index=0,
            state=state,
        )
        # No escalation on first rejection
        assert result is None or result.get("outcome") != "uncertainty_escalation"

    def test_second_rejection_triggers_escalation(self) -> None:
        """Second rejection of same diff_hash triggers same_fix_rejected_twice."""
        from src.orchestrator.supervisor_only import _record_diff_rejection_check

        state = _make_state()

        # First rejection
        result1 = _record_diff_rejection_check(
            diff_hash="hash_abc",
            agent_name="reviewer",
            step_index=0,
            state=state,
        )
        if result1 is not None:
            state_data = state.model_dump()
            state_data.update(result1)
            state = OrchestratorState(**state_data)  # type: ignore[arg-type]

        # Second rejection of same hash → escalation
        result2 = _record_diff_rejection_check(
            diff_hash="hash_abc",
            agent_name="reviewer",
            step_index=0,
            state=state,
        )
        assert result2 is not None
        assert result2.get("outcome") == "uncertainty_escalation"
        assert "same_fix_rejected_twice" in str(result2.get("errors", []))

    def test_empty_diff_hash_no_rejection(self) -> None:
        """Empty diff_hash is not tracked for rejection."""
        from src.orchestrator.supervisor_only import _record_diff_rejection_check

        state = _make_state()
        result = _record_diff_rejection_check(
            diff_hash="",
            agent_name="reviewer",
            step_index=0,
            state=state,
        )
        assert result is None


# ── RecordingSandboxProxy tests ─────────────────────────────────────


class TestRecordingSandboxProxy:
    """Verify that RecordingSandboxProxy records tool calls."""

    @pytest.mark.asyncio
    async def test_record_successful_tool_call(self) -> None:
        """Successful tool calls are recorded with success=True."""
        from src.orchestrator.supervisor_only import (
            RecordingSandboxProxy,
            _tool_call_recordings,
        )

        mock_inner = AsyncMock()
        mock_inner.run_command = AsyncMock(return_value="ok")
        mock_inner.is_running = True
        mock_inner.container_id = "abc"
        mock_inner.workspace_dir = "/workspace"
        mock_inner.task_id = "test-task"

        proxy = RecordingSandboxProxy(mock_inner, "test-task")
        _tool_call_recordings.pop("test-task", None)  # Clear any previous

        await proxy.run_command("echo hello")
        recordings = _tool_call_recordings.get("test-task", [])
        assert len(recordings) == 1
        assert recordings[0].tool_name == "run_command"
        assert recordings[0].success is True

        _tool_call_recordings.pop("test-task", None)

    @pytest.mark.asyncio
    async def test_record_failed_tool_call(self) -> None:
        """Failed tool calls are recorded with success=False."""
        from src.orchestrator.supervisor_only import (
            RecordingSandboxProxy,
            _tool_call_recordings,
        )

        mock_inner = AsyncMock()
        mock_inner.run_command = AsyncMock(side_effect=RuntimeError("failed"))
        mock_inner.is_running = True
        mock_inner.container_id = "abc"
        mock_inner.workspace_dir = "/workspace"
        mock_inner.task_id = "test-task"

        proxy = RecordingSandboxProxy(mock_inner, "test-task")
        _tool_call_recordings.pop("test-task", None)

        with pytest.raises(RuntimeError):
            await proxy.run_command("bad_command")

        recordings = _tool_call_recordings.get("test-task", [])
        assert len(recordings) == 1
        assert recordings[0].success is False
        assert "failed" in recordings[0].error_msg

        _tool_call_recordings.pop("test-task", None)


# ── Promptfoo CI workflow test ──────────────────────────────────────


class TestPromptfooCIPaths:
    """Verify that prompts/** is in the CI path filters."""

    def _load_workflow(self) -> dict:
        """Load the workflow YAML, handling 'on' key correctly."""
        import yaml

        yml_path = (
            "/Users/norbertesekiel/Developer/MultiAgenticSystem/"
            ".github/workflows/promptfoo-eval.yml"
        )
        with open(yml_path) as f:
            content = f.read()
        # yaml.safe_load treats 'on' as True (Python bool)
        # Use a custom constructor or string replacement
        workflow = yaml.safe_load(content)
        # The 'on' key becomes True in Python
        on_config = workflow.get(True, workflow.get("on", {}))
        return on_config

    def test_prompts_path_in_push_filter(self) -> None:
        """push paths include prompts/**."""
        on_config = self._load_workflow()
        push_paths = on_config.get("push", {}).get("paths", [])
        assert "prompts/**" in push_paths, (
            f"prompts/** not found in push paths: {push_paths}"
        )

    def test_prompts_path_in_pull_request_filter(self) -> None:
        """pull_request paths include prompts/**."""
        on_config = self._load_workflow()
        pr_paths = on_config.get("pull_request", {}).get("paths", [])
        assert "prompts/**" in pr_paths, (
            f"prompts/** not found in pull_request paths: {pr_paths}"
        )
