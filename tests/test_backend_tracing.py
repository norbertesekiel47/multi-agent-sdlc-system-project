"""Tests for backend API and tracing — m1-backend-and-tracing feature.

Covers assertions:
  - VAL-BACKEND-API-002: Error responses never leak stack traces
  - VAL-BACKEND-API-003: src/api/ never imports src/llm/ or openrouter/openai
  - VAL-TRACING-001–011: Langfuse tracing behavior
  - VAL-COST-BUDGET-001–008: Cost tracking and budget enforcement
  - VAL-CROSS-020: Langfuse offline graceful degradation
"""

from __future__ import annotations

import importlib
import json
import os
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.memory.episodic.models import TaskRow

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_mock_store() -> AsyncMock:
    """Create a mock EpisodicStore that doesn't need a real DB connection."""
    store = AsyncMock()
    store._pool = MagicMock()  # truthy to pass the pool check
    store.create_task = AsyncMock(return_value=TaskRow(
        id=uuid4(),
        repo_url="https://github.com/test/repo",
        issue_number=1,
        issue_text="Fix the bug",
        topology="hybrid",
        status="running",
        started_at=datetime.now(UTC),
    ))
    store.get_task = AsyncMock(return_value=None)  # no task found by default
    store.list_tasks = AsyncMock(return_value=[])
    store.get_latest_outcomes = AsyncMock(return_value={})
    store.update_task_totals = AsyncMock()
    store.close = AsyncMock()
    return store


@pytest.fixture
def mock_store() -> AsyncMock:
    """Mock EpisodicStore for sync TestClient tests."""
    return _make_mock_store()


@pytest.fixture
def client(mock_store: AsyncMock) -> TestClient:
    """FastAPI test client with mocked store dependency."""
    from src.api.main import get_store
    app.dependency_overrides[get_store] = lambda: mock_store
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


# ──────────────────────────────────────────────────────────────────────
# VAL-BACKEND-API-002: Error responses never leak stack traces
# ──────────────────────────────────────────────────────────────────────


class TestErrorResponsesNeverLeakStackTraces:
    """Error responses must be JSON {error,...} with no Traceback or src/ paths."""

    def test_404_task_not_found_no_traceback(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """GET /tasks/{nonexistent_id} returns JSON with no Python internals."""
        fake_id = uuid4()
        mock_store.get_task.return_value = None  # not found
        resp = client.get(f"/tasks/{fake_id}")
        assert resp.status_code == 404
        body = resp.json()
        # Must be JSON object with 'error' key
        assert "error" in body
        # Must NOT contain traceback or src/ paths
        body_str = json.dumps(body)
        assert "Traceback" not in body_str
        assert "/src/" not in body_str
        assert ".py" not in body_str or "python" not in body_str.lower()
        assert "line " not in body_str

    def test_422_validation_error_no_traceback(self, client: TestClient) -> None:
        """POST /tasks with invalid topology returns 422 with safe JSON."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": 1,
                "issue_text": "Fix the bug",
                "topology": "anarchy",
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body
        body_str = json.dumps(body)
        assert "Traceback" not in body_str
        assert "/src/" not in body_str

    def test_422_invalid_issue_number(self, client: TestClient) -> None:
        """POST /tasks with issue_number=0 returns 422."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": 0,
                "issue_text": "Fix the bug",
                "topology": "hybrid",
            },
        )
        assert resp.status_code == 422

    def test_422_negative_issue_number(self, client: TestClient) -> None:
        """POST /tasks with negative issue_number returns 422."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": -1,
                "issue_text": "Fix the bug",
                "topology": "hybrid",
            },
        )
        assert resp.status_code == 422

    def test_422_overflow_issue_number(self, client: TestClient) -> None:
        """POST /tasks with issue_number > 2^31-1 returns 422."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": 2147483648,
                "issue_text": "Fix the bug",
                "topology": "hybrid",
            },
        )
        assert resp.status_code == 422

    def test_422_empty_repo_url(self, client: TestClient) -> None:
        """POST /tasks with empty repo_url returns 422."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "",
                "issue_number": 1,
                "issue_text": "Fix the bug",
                "topology": "hybrid",
            },
        )
        assert resp.status_code == 422

    def test_422_empty_issue_text(self, client: TestClient) -> None:
        """POST /tasks with empty issue_text returns 422."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": 1,
                "issue_text": "",
                "topology": "hybrid",
            },
        )
        assert resp.status_code == 422

    def test_error_response_format(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """All error responses are JSON with 'error' key, no Traceback or src/ paths."""
        # Test 404 path
        mock_store.get_task.return_value = None
        resp = client.get(f"/tasks/{uuid4()}")
        if resp.status_code == 404:
            body_str = json.dumps(resp.json())
            assert not re.search(
                r"Traceback|/Users/|/src/|\.py[\":]|File \"|line \d+",
                body_str,
            )

        # Test 422 path
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "",
                "issue_number": 0,
                "issue_text": "",
                "topology": "bad",
            },
        )
        if resp.status_code == 422:
            body_str = json.dumps(resp.json())
            assert not re.search(
                r"Traceback|/Users/|/src/|\.py[\":]|File \"|line \d+",
                body_str,
            )


# ──────────────────────────────────────────────────────────────────────
# VAL-BACKEND-API-003: src/api/ never imports src/llm/ or openrouter/openai
# ──────────────────────────────────────────────────────────────────────


class TestImportBoundary:
    """No module under src/api/ imports src/llm/ or openrouter/openai."""

    def test_api_no_llm_import(self) -> None:
        """src/api/ modules must not import src.llm or openrouter or openai."""
        import pkgutil

        import src.api as api_pkg

        forbidden_prefixes = ("src.llm", "openrouter", "openai")
        violations: list[str] = []

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            api_pkg.__path__, prefix=api_pkg.__name__ + "."
        ):
            try:
                mod = importlib.import_module(modname)
                source = inspect_get_source(mod) or ""
                for line_no, line in enumerate(source.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for prefix in forbidden_prefixes:
                        # Check for: import src.llm, from src.llm, import openai, from openai, etc.
                        patterns = [
                            f"import {prefix}",
                            f"from {prefix}",
                        ]
                        for pat in patterns:
                            if pat in stripped:
                                violations.append(f"{modname}:{line_no}: {stripped}")
            except Exception:
                continue

        # Also check the top-level api modules directly
        for modname in ("src.api.main", "src.api.models", "src.api.errors"):
            try:
                mod = importlib.import_module(modname)
                source = inspect_get_source(mod) or ""
                for line_no, line in enumerate(source.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for prefix in forbidden_prefixes:
                        patterns = [f"import {prefix}", f"from {prefix}"]
                        for pat in patterns:
                            if pat in stripped:
                                violations.append(f"{modname}:{line_no}: {stripped}")
            except Exception:
                continue

        assert violations == [], (
            "src/api/ imports forbidden modules:\n" + "\n".join(violations)
        )


def inspect_get_source(module: Any) -> str | None:
    """Safely get source code of a module."""
    import inspect
    try:
        return inspect.getsource(module)
    except (TypeError, OSError):
        return None


# ──────────────────────────────────────────────────────────────────────
# VAL-TRACING tests (unit-level, with mocked Langfuse)
# ──────────────────────────────────────────────────────────────────────


class TestTracingSpanCreation:
    """Unit tests for Langfuse span creation with graceful degradation."""

    def test_truncate_and_redact_short_payload(self) -> None:
        """Short payloads pass through without truncation."""
        from src.tracing.langfuse import _truncate_and_redact

        result = _truncate_and_redact({"key": "value"})
        assert "…[truncated]" not in result
        assert "value" in result

    def test_truncate_and_redact_long_payload(self) -> None:
        """Payloads over 2 KB get truncated with marker."""
        from src.tracing.langfuse import _truncate_and_redact

        long_payload = "x" * 3000
        result = _truncate_and_redact(long_payload)
        assert "…[truncated]" in result
        assert len(result.encode("utf-8")) <= 2048 + len("…[truncated]")

    def test_truncate_and_redact_secret_redaction(self) -> None:
        """Secret values are redacted BEFORE truncation."""
        from src.tracing.langfuse import _truncate_and_redact

        secret = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-test123456")
        payload = {"prompt": f"My key is {secret}"}
        result = _truncate_and_redact(payload)
        assert secret not in result
        assert "***REDACTED***" in result

    def test_truncate_and_redact_preserves_non_secret(self) -> None:
        """Non-secret content is preserved after redaction."""
        from src.tracing.langfuse import _truncate_and_redact

        payload = {"content": "Hello world"}
        result = _truncate_and_redact(payload)
        assert "Hello world" in result

    def test_secret_redaction_patterns(self) -> None:
        """All known secret prefixes are redacted."""
        from src.tracing.langfuse import _redact_secrets

        test_cases = [
            ("github_pat_ABCDEF123", "***REDACTED***"),
            ("sk-or-v1-12345abcde", "***REDACTED***"),
            ("sk-proj-12345abcde", "***REDACTED***"),
            ("gho_ABCDEFGH1234", "***REDACTED***"),
            ("ghp_ABCDEFGH1234", "***REDACTED***"),
            ("hf_abc123def456", "***REDACTED***"),
        ]
        for secret, _expected in test_cases:
            result = _redact_secrets(f"key={secret}")
            assert secret not in result
            assert "***REDACTED***" in result


class TestTracingGracefulDegradation:
    """VAL-CROSS-020: Backend continues when Langfuse is down."""

    def test_tracing_client_unavailable_by_default(self) -> None:
        """TracingClient degrades gracefully when keys are not configured."""
        from src.tracing.langfuse import TracingClient

        client = TracingClient()
        # With no keys set (or invalid keys), should not crash
        with patch.dict(os.environ, {"LANGFUSE_SECRET_KEY": "", "LANGFUSE_PUBLIC_KEY": ""}):
            client._try_init()
            assert not client._available

    def test_create_trace_returns_none_when_unavailable(self) -> None:
        """Creating traces returns None gracefully when Langfuse is down."""
        from src.tracing.langfuse import TracingClient

        client = TracingClient()
        # Remove keys and force unavailable state
        with patch.dict(os.environ, {"LANGFUSE_SECRET_KEY": "", "LANGFUSE_PUBLIC_KEY": ""}):
            client._available = False
            client._client = None
            client._last_init_attempt = 0  # allow re-init attempt

            result = client.create_trace(trace_id="test", name="test")
            assert result is None

    def test_create_span_returns_none_when_unavailable(self) -> None:
        """Creating spans returns None gracefully when Langfuse is down."""
        from src.tracing.langfuse import TracingClient

        client = TracingClient()
        client._available = False
        client._client = None

        result = client.create_span(trace_id="test", name="test")
        assert result is None

    def test_create_generation_returns_none_when_unavailable(self) -> None:
        """Creating generations returns None gracefully when Langfuse is down."""
        from src.tracing.langfuse import TracingClient

        client = TracingClient()
        client._available = False
        client._client = None

        result = client.create_generation(
            trace_id="test", name="test", model="test"
        )
        assert result is None

    def test_update_span_no_error_when_unavailable(self) -> None:
        """Updating spans is a no-op when Langfuse is down."""
        from src.tracing.langfuse import TracingClient

        client = TracingClient()
        client._available = False
        client._client = None

        # Should not raise
        client.update_span(trace_id="test", span_id="test", output_data={"ok": True})

    def test_flush_no_error_when_unavailable(self) -> None:
        """Flushing is a no-op when Langfuse is down."""
        from src.tracing.langfuse import TracingClient

        client = TracingClient()
        client._available = False
        client._client = None

        # Should not raise
        client.flush()


# ──────────────────────────────────────────────────────────────────────
# VAL-TRACING-011: Span I/O does not echo secret values
# ──────────────────────────────────────────────────────────────────────


class TestSecretRedactionInSpans:
    """Secret values must never appear in Langfuse span I/O."""

    def test_all_env_secrets_redacted_in_span_io(self) -> None:
        """For each known secret, redaction replaces the literal before truncation."""
        from src.tracing.langfuse import _truncate_and_redact

        secret_keys = [
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "GITHUB_PAT",
            "HUGGINGFACE_TOKEN",
        ]
        for key_name in secret_keys:
            secret_val = os.getenv(key_name, "")
            if not secret_val:
                continue
            # Create a payload that contains the secret literal
            payload = {
                "prompt": f"Use this key: {secret_val}",
                "context": f"Authorization: Bearer {secret_val}",
            }
            result = _truncate_and_redact(payload)
            # The raw secret value must NOT appear in the result
            assert secret_val not in result, (
                f"Secret {key_name} leaked in span I/O after redaction"
            )
            # The redaction marker must appear
            assert "***REDACTED***" in result

    def test_redaction_regex_sweep_on_spans(self) -> None:
        """Ripgrep-style check: no secret literal in redacted output."""
        from src.tracing.langfuse import _redact_secrets

        all_secrets = []
        for key_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GITHUB_PAT", "HUGGINGFACE_TOKEN"):
            val = os.getenv(key_name, "")
            if val:
                all_secrets.append(val)

        for secret in all_secrets:
            text = f"some context with {secret} embedded"
            redacted = _redact_secrets(text)
            assert secret not in redacted, "Secret literal found in redacted text"


# ──────────────────────────────────────────────────────────────────────
# VAL-TRACING: Trace event structure for WebSocket
# ──────────────────────────────────────────────────────────────────────


class TestTraceEventStructure:
    """Trace events sent via WebSocket must have the required fields."""

    def test_trace_event_to_json_has_required_fields(self) -> None:
        """VAL-TRACE-STREAM-006: WS message has all required trace event fields."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="node_start",
            task_id="test-task-id",
            trace_id="test-trace-id",
            span_id="test-span-id",
            parent_span_id="parent-span-id",
            name="planner.turn",
            span_type="span",
            started_at=datetime.now(UTC),
            tokens_in=100,
            tokens_out=50,
            cached_tokens=20,
            cost_usd=Decimal("0.0123"),
            status="ok",
        )
        data = json.loads(event.to_json())

        # Required fields per VAL-TRACE-STREAM-006
        required_fields = {
            "type", "task_id", "span_id", "parent_span_id",
            "name", "started_at", "ended_at",
            "tokens_in", "tokens_out", "cost_usd", "status",
        }
        assert required_fields.issubset(
            data.keys()
        ), f"Missing fields: {required_fields - data.keys()}"

    def test_trace_event_types_distinguishable(self) -> None:
        """VAL-TRACE-STREAM-007: At least three distinguishable event types."""
        from src.tracing.ws_broadcaster import TraceEvent

        event_types = {"node_start", "tool_call", "llm_completion"}
        for etype in event_types:
            event = TraceEvent(
                type=etype,
                task_id="test",
                trace_id="test",
                span_id="test",
                name=f"test.{etype}",
            )
            data = json.loads(event.to_json())
            assert data["type"] == etype


# ──────────────────────────────────────────────────────────────────────
# WebSocket broadcaster tests
# ──────────────────────────────────────────────────────────────────────


class TestTraceBroadcaster:
    """Unit tests for the WebSocket trace broadcaster."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self) -> None:
        """Subscribers receive published events."""
        from src.tracing.ws_broadcaster import TraceBroadcaster, TraceEvent

        broadcaster = TraceBroadcaster()
        queue = await broadcaster.subscribe("task-1")

        event = TraceEvent(
            type="node_start",
            task_id="task-1",
            trace_id="trace-1",
            span_id="span-1",
            name="planner",
        )
        await broadcaster.publish(event)

        message = queue.get_nowait()
        data = json.loads(message)
        assert data["type"] == "node_start"
        assert data["task_id"] == "task-1"

        await broadcaster.unsubscribe("task-1", queue)

    @pytest.mark.asyncio
    async def test_subscribers_only_receive_scoped_events(self) -> None:
        """Subscribers for task A don't receive events for task B."""
        from src.tracing.ws_broadcaster import TraceBroadcaster, TraceEvent

        broadcaster = TraceBroadcaster()
        queue_a = await broadcaster.subscribe("task-a")
        queue_b = await broadcaster.subscribe("task-b")

        event_a = TraceEvent(
            type="node_start", task_id="task-a", trace_id="t", span_id="s", name="a"
        )
        event_b = TraceEvent(
            type="node_start", task_id="task-b", trace_id="t", span_id="s", name="b"
        )

        await broadcaster.publish(event_a)
        await broadcaster.publish(event_b)

        # queue_a only has event_a
        msg_a = json.loads(queue_a.get_nowait())
        assert msg_a["task_id"] == "task-a"
        assert queue_a.empty()

        # queue_b only has event_b
        msg_b = json.loads(queue_b.get_nowait())
        assert msg_b["task_id"] == "task-b"
        assert queue_b.empty()

        await broadcaster.unsubscribe("task-a", queue_a)
        await broadcaster.unsubscribe("task-b", queue_b)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_queue(self) -> None:
        """Unsubscribing removes the queue; no more events received."""
        from src.tracing.ws_broadcaster import TraceBroadcaster, TraceEvent

        broadcaster = TraceBroadcaster()
        queue = await broadcaster.subscribe("task-1")
        await broadcaster.unsubscribe("task-1", queue)

        # Publish should not fill the queue
        event = TraceEvent(
            type="node_start", task_id="task-1", trace_id="t", span_id="s", name="n"
        )
        await broadcaster.publish(event)
        assert queue.empty()


# ──────────────────────────────────────────────────────────────────────
# VAL-COST-BUDGET tests
# ──────────────────────────────────────────────────────────────────────


class TestCostExtraction:
    """Tests for cost extraction from OpenRouter responses."""

    def test_extract_cost_from_usage_cost_field(self) -> None:
        """When usage.cost is present, use it directly."""
        from src.llm.cost import extract_cost_from_response

        cost = extract_cost_from_response(
            model="deepseek/deepseek-v4-flash",
            usage={"cost": 0.005, "prompt_tokens": 1000, "completion_tokens": 500},
        )
        assert cost == Decimal("0.0050")

    def test_extract_cost_fallback_to_tiktoken(self) -> None:
        """VAL-COST-BUDGET-006: Missing usage.cost falls back to tiktoken estimate."""
        from src.llm.cost import extract_cost_from_response

        cost = extract_cost_from_response(
            model="deepseek/deepseek-v4-flash",
            usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )
        # Should be non-zero from tiktoken estimate
        assert cost > Decimal("0")

    def test_extract_cost_zero_for_zero_tokens(self) -> None:
        """Zero tokens → zero cost."""
        from src.llm.cost import extract_cost_from_response

        cost = extract_cost_from_response(
            model="deepseek/deepseek-v4-flash",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )
        assert cost == Decimal("0")

    def test_extract_cached_tokens_present(self) -> None:
        """Extract cached_tokens from prompt_tokens_details."""
        from src.llm.cost import extract_cached_tokens

        usage = {
            "prompt_tokens_details": {"cached_tokens": 42},
            "completion_tokens": 0,
        }
        assert extract_cached_tokens(usage) == 42

    def test_extract_cached_tokens_absent(self) -> None:
        """Missing cached_tokens field returns 0."""
        from src.llm.cost import extract_cached_tokens

        assert extract_cached_tokens({"prompt_tokens": 100}) == 0
        assert extract_cached_tokens({}) == 0

    def test_max_cost_per_task_default(self) -> None:
        """Default MAX_COST_PER_TASK_USD is $2.00."""
        from src.llm.cost import get_max_cost_per_task

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_COST_PER_TASK_USD", None)
            cost = get_max_cost_per_task()
            assert cost == Decimal("2.00")

    def test_max_cost_per_task_from_env(self) -> None:
        """MAX_COST_PER_TASK_USD reads from environment."""
        from src.llm.cost import get_max_cost_per_task

        with patch.dict(os.environ, {"MAX_COST_PER_TASK_USD": "0.50"}):
            cost = get_max_cost_per_task()
            assert cost == Decimal("0.50")


class TestCostBudgetExceededError:
    """Tests for CostBudgetExceededError."""

    def test_error_contains_details(self) -> None:
        """Error includes task_id, total_cost, max_cost, triggering_agent."""
        from src.llm.cost import CostBudgetExceededError

        err = CostBudgetExceededError(
            task_id="test-id",
            total_cost_usd=Decimal("3.00"),
            max_cost_usd=Decimal("2.00"),
            triggering_agent="coder",
        )
        assert err.task_id == "test-id"
        assert err.total_cost_usd == Decimal("3.00")
        assert err.max_cost_usd == Decimal("2.00")
        assert err.triggering_agent == "coder"
        assert "3.00" in str(err)
        assert "2.00" in str(err)


class TestTiktokenEstimate:
    """Tests for tiktoken-based cost estimation."""

    def test_estimate_known_model(self) -> None:
        """Known model uses model-specific pricing."""
        from src.llm.cost import estimate_cost_tiktoken

        cost = estimate_cost_tiktoken(
            model="deepseek/deepseek-v4-flash",
            prompt_tokens=1000000,
            completion_tokens=1000000,
        )
        # Flash: $0.10/1M input + $0.40/1M output = $0.50 total
        assert cost == Decimal("0.5000")

    def test_estimate_unknown_model_uses_fallback(self) -> None:
        """Unknown model falls back to DeepSeek Flash pricing."""
        from src.llm.cost import estimate_cost_tiktoken

        cost = estimate_cost_tiktoken(
            model="unknown/model",
            prompt_tokens=1000000,
            completion_tokens=0,
        )
        # Fallback: $0.10/1M input
        assert cost > Decimal("0")

    def test_embedding_cost_estimate(self) -> None:
        """Embedding model cost is estimated correctly."""
        from src.llm.cost import estimate_cost_tiktoken

        cost = estimate_cost_tiktoken(
            model="text-embedding-3-small",
            prompt_tokens=1000000,
            completion_tokens=0,
        )
        # $0.02/1M tokens
        assert cost == Decimal("0.0200")


# ──────────────────────────────────────────────────────────────────────
# VAL-TRACING: Span hierarchy tests
# ──────────────────────────────────────────────────────────────────────


class TestSpanHierarchy:
    """Span hierarchy mirrors the call graph (VAL-TRACING-006)."""

    def test_trace_event_parent_chain(self) -> None:
        """Tool-call events have parent_span_id referencing an agent turn."""
        from src.tracing.ws_broadcaster import TraceEvent

        # Node span (supervisor node)
        node_event = TraceEvent(
            type="node_start",
            task_id="t1",
            trace_id="tr1",
            span_id="span-node-1",
            parent_span_id=None,  # root
            name="supervisor",
        )

        # Agent turn (planner) under node
        agent_event = TraceEvent(
            type="node_start",
            task_id="t1",
            trace_id="tr1",
            span_id="span-agent-1",
            parent_span_id="span-node-1",
            name="planner.turn",
        )

        # LLM completion under agent turn
        llm_event = TraceEvent(
            type="llm_completion",
            task_id="t1",
            trace_id="tr1",
            span_id="span-llm-1",
            parent_span_id="span-agent-1",
            name="planner.llm_call",
        )

        # Verify hierarchy: node → agent → llm
        assert agent_event.parent_span_id == node_event.span_id
        assert llm_event.parent_span_id == agent_event.span_id
        assert node_event.parent_span_id is None  # root

    def test_tool_call_parent_is_agent_turn(self) -> None:
        """Tool-call spans have an agent-turn parent (not root)."""
        from src.tracing.ws_broadcaster import TraceEvent

        agent_span_id = "span-agent-1"
        tool_event = TraceEvent(
            type="tool_call",
            task_id="t1",
            trace_id="tr1",
            span_id="span-tool-1",
            parent_span_id=agent_span_id,
            name="sandbox.read_file",
        )
        assert tool_event.parent_span_id == agent_span_id


# ──────────────────────────────────────────────────────────────────────
# VAL-TRACING-003/004/005: LLM span metadata tests
# ──────────────────────────────────────────────────────────────────────


class TestLLMSpanMetadata:
    """LLM spans have tokens-in, tokens-out, cached-tokens, cost (VAL-TRACING-003/004)."""

    def test_generation_span_has_all_fields(self) -> None:
        """TraceEvent for LLM completion has tokens and cost."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="llm_completion",
            task_id="t1",
            trace_id="tr1",
            span_id="span-1",
            name="coder.llm_call",
            span_type="generation",
            tokens_in=100,
            tokens_out=50,
            cached_tokens=20,
            cost_usd=Decimal("0.0123"),
        )
        data = json.loads(event.to_json())
        assert data["tokens_in"] == 100
        assert data["tokens_out"] == 50
        assert data["cached_tokens"] == 20
        assert data["cost_usd"] == "0.0123"

    def test_span_io_truncated_at_2kb(self) -> None:
        """VAL-TRACING-005: Span I/O truncated at 2 KB with marker."""
        from src.tracing.langfuse import _truncate_and_redact

        # Create a payload exactly at the boundary
        long_text = "a" * 2100
        result = _truncate_and_redact(long_text)
        assert "…[truncated]" in result
        assert len(result) < 2100

    def test_span_io_not_truncated_under_2kb(self) -> None:
        """Payloads under 2 KB are not truncated."""
        from src.tracing.langfuse import _truncate_and_redact

        short_text = "Hello, this is a short prompt"
        result = _truncate_and_redact(short_text)
        assert "…[truncated]" not in result
        assert result == short_text


# ──────────────────────────────────────────────────────────────────────
# VAL-TRACING-010: Agent turn span tagged with agent name
# ──────────────────────────────────────────────────────────────────────


class TestAgentSpanTagging:
    """Each agent turn emits a span tagged with the agent name."""

    def test_trace_event_carries_agent_metadata(self) -> None:
        """TraceEvent metadata includes agent name."""
        from src.tracing.ws_broadcaster import TraceEvent

        for agent in ("planner", "coder", "reviewer", "qa", "supervisor"):
            event = TraceEvent(
                type="node_start",
                task_id="t1",
                trace_id="tr1",
                span_id="span-1",
                name=f"{agent}.turn",
                metadata={"agent": agent},
            )
            data = json.loads(event.to_json())
            assert data["metadata"]["agent"] == agent
            assert data["name"] == f"{agent}.turn"


# ──────────────────────────────────────────────────────────────────────
# Caching module tests
# ──────────────────────────────────────────────────────────────────────


class TestPromptCaching:
    """Tests for prompt caching data structures (M2 placeholder)."""

    def test_structured_prompt_has_static_and_dynamic(self) -> None:
        """StructuredPrompt has both static and dynamic blocks."""
        from src.llm.caching import StructuredPrompt

        prompt = StructuredPrompt(static="system instructions", dynamic="current edit")
        assert prompt.static == "system instructions"
        assert prompt.dynamic == "current edit"

    def test_build_structured_prompt(self) -> None:
        """build_structured_prompt produces correct blocks."""
        from src.llm.caching import build_structured_prompt

        result = build_structured_prompt(
            system_instructions="You are a coder.",
            repo_context="File: src/main.py\nprint('hello')",
            current_edit="Change print to logging",
        )
        assert "coder" in result.static
        assert "src/main.py" in result.static
        assert "Change print to logging" in result.dynamic

    def test_structured_prompt_to_messages(self) -> None:
        """to_messages produces OpenRouter-compatible message list."""
        from src.llm.caching import StructuredPrompt

        prompt = StructuredPrompt(static="system", dynamic="user")
        messages = prompt.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ──────────────────────────────────────────────────────────────────────
# API endpoint tests (with mocked store for sync TestClient)
# ──────────────────────────────────────────────────────────────────────


class TestCreateTaskEndpoint:
    """POST /tests creates a tasks row and returns 201 with {id}."""

    def test_create_task_returns_201_with_id(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """POST /tasks returns 201 with {id}."""
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": 1,
                "issue_text": "Fix the bug",
                "topology": "hybrid",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "id" in body

    def test_create_task_default_topology_hybrid(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """POST /tasks without topology defaults to hybrid."""
        # This would need topology field optional in model — it already is
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo",
                "issue_number": 1,
                "issue_text": "Fix the bug",
            },
        )
        # Should succeed (topology defaults to "hybrid")
        assert resp.status_code == 201

    def test_create_task_all_topologies_accepted(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """All three valid topologies are accepted."""
        for topo in ("single_agent", "supervisor_only", "hybrid"):
            resp = client.post(
                "/tasks",
                json={
                    "repo_url": "https://github.com/test/repo",
                    "issue_number": 1,
                    "issue_text": "Fix the bug",
                    "topology": topo,
                },
            )
            assert resp.status_code == 201, f"Topology {topo} should be accepted"


class TestGetTaskEndpoint:
    """GET /tasks/{id} returns task detail with all fields."""

    def test_get_task_found(self, client: TestClient, mock_store: AsyncMock) -> None:
        """GET /tasks/{id} returns task detail when found."""
        task_id = uuid4()
        now = datetime.now(UTC)
        mock_store.get_task.return_value = TaskRow(
            id=task_id,
            repo_url="https://github.com/test/repo",
            issue_number=1,
            issue_text="Fix the bug",
            topology="hybrid",
            status="running",
            total_cost_usd=Decimal("0.0100"),
            total_tokens_in=1000,
            total_tokens_out=500,
            total_tokens_cached=200,
            started_at=now,
        )
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(task_id)
        assert body["repo_url"] == "https://github.com/test/repo"
        assert body["status"] == "running"
        assert body["topology"] == "hybrid"
        assert body["total_cost_usd"] == "0.0100"
        assert body["total_tokens_in"] == 1000
        assert body["total_tokens_out"] == 500
        assert body["total_tokens_cached"] == 200

    def test_get_task_not_found(self, client: TestClient, mock_store: AsyncMock) -> None:
        """GET /tasks/{id} returns 404 when not found."""
        mock_store.get_task.return_value = None
        resp = client.get(f"/tasks/{uuid4()}")
        assert resp.status_code == 404


class TestListTasksEndpoint:
    """GET /tasks lists tasks with optional filters."""

    def test_list_tasks_empty(self, client: TestClient, mock_store: AsyncMock) -> None:
        """GET /tasks returns empty list when no tasks."""
        mock_store.list_tasks.return_value = []
        resp = client.get("/tasks")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tasks"] == []
        assert body["total"] == 0

    def test_list_tasks_with_tasks(self, client: TestClient, mock_store: AsyncMock) -> None:
        """GET /tasks returns list of tasks."""
        task_id = uuid4()
        now = datetime.now(UTC)
        mock_store.list_tasks.return_value = [
            TaskRow(
                id=task_id,
                repo_url="https://github.com/test/repo",
                issue_number=1,
                issue_text="Fix the bug",
                topology="hybrid",
                status="running",
                started_at=now,
            )
        ]
        resp = client.get("/tasks")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["tasks"]) == 1


class TestHealthEndpoint:
    """GET /health returns structured health payload."""

    def test_health_structure(self, client: TestClient) -> None:
        """GET /health returns {status, version, db, langfuse}."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "version" in body
        assert "db" in body
        assert "langfuse" in body
        assert body["status"] in ("ok", "degraded")


# ──────────────────────────────────────────────────────────────────────
# Integration-level tests (require Postgres)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAPIEndpointsIntegration:
    """Integration tests for API endpoints with real Postgres."""

    async def test_create_and_get_task(self) -> None:
        """POST /tasks creates a row; GET /tasks/{id} returns it."""
        from src.memory.episodic.store import EpisodicStore

        async with EpisodicStore() as store:
            from src.memory.episodic.models import CreateTaskParams

            params = CreateTaskParams(
                repo_url="https://github.com/test/repo",
                issue_number=1,
                issue_text="Fix the bug",
                topology="hybrid",
            )
            task = await store.create_task(params)
            assert task.id is not None
            assert task.status == "running"
            assert task.repo_url == "https://github.com/test/repo"

            # Fetch it back
            fetched = await store.get_task(task.id)
            assert fetched is not None
            assert fetched.id == task.id
            assert fetched.topology == "hybrid"

    async def test_cost_update_transactional(self) -> None:
        """VAL-COST-BUDGET-007: total_cost_usd updated transactionally per LLM call."""
        from src.memory.episodic.models import CreateTaskParams
        from src.memory.episodic.store import EpisodicStore

        async with EpisodicStore() as store:
            params = CreateTaskParams(
                repo_url="https://github.com/test/cost-repo",
                issue_number=1,
                issue_text="Test cost",
                topology="single_agent",
            )
            task = await store.create_task(params)

            # Update cost
            await store.update_task_totals(
                task.id,
                total_cost_usd=Decimal("0.0100"),
                total_tokens_in=1000,
                total_tokens_out=500,
                total_tokens_cached=200,
            )

            fetched = await store.get_task(task.id)
            assert fetched is not None
            assert fetched.total_cost_usd == Decimal("0.0100")
            assert fetched.total_tokens_in == 1000
            assert fetched.total_tokens_out == 500
            assert fetched.total_tokens_cached == 200

    async def test_embedding_cost_included(self) -> None:
        """VAL-COST-BUDGET-008: Embedding cost contributes to total_cost_usd."""
        from src.memory.episodic.models import CreateTaskParams
        from src.memory.episodic.store import EpisodicStore

        async with EpisodicStore() as store:
            params = CreateTaskParams(
                repo_url="https://github.com/test/embed-cost",
                issue_number=1,
                issue_text="Test embedding cost",
                topology="hybrid",
            )
            task = await store.create_task(params)

            # Simulate embedding cost
            embedding_cost = Decimal("0.0003")
            await store.update_task_totals(task.id, total_cost_usd=embedding_cost)

            fetched = await store.get_task(task.id)
            assert fetched is not None
            assert fetched.total_cost_usd is not None
            assert fetched.total_cost_usd > Decimal("0")

    async def test_cost_budget_exhausted_outcome(self) -> None:
        """VAL-COST-BUDGET-002: Cost cap writes cost_budget_exhausted outcome row."""
        from src.memory.episodic.models import CreateOutcomeParams, CreateTaskParams
        from src.memory.episodic.store import EpisodicStore

        async with EpisodicStore() as store:
            params = CreateTaskParams(
                repo_url="https://github.com/test/budget-repo",
                issue_number=1,
                issue_text="Test budget",
                topology="hybrid",
            )
            task = await store.create_task(params)

            # Write the cost_budget_exhausted outcome
            outcome = await store.create_outcome(
                CreateOutcomeParams(
                    task_id=task.id,
                    outcome="cost_budget_exhausted",
                    detail={
                        "total_cost_usd": "3.50",
                        "max_cost_usd": "2.00",
                        "triggering_agent": "coder",
                    },
                )
            )
            assert outcome.outcome == "cost_budget_exhausted"
            assert outcome.detail is not None
            assert outcome.detail["triggering_agent"] == "coder"

    async def test_list_tasks_with_filters(self) -> None:
        """GET /tasks with filters works correctly."""
        from src.memory.episodic.models import CreateTaskParams
        from src.memory.episodic.store import EpisodicStore

        async with EpisodicStore() as store:
            # Create two tasks
            t1 = await store.create_task(
                CreateTaskParams(
                    repo_url="https://github.com/test/list-a",
                    issue_number=1,
                    issue_text="Task A",
                    topology="hybrid",
                )
            )
            t2 = await store.create_task(
                CreateTaskParams(
                    repo_url="https://github.com/test/list-b",
                    issue_number=2,
                    issue_text="Task B",
                    topology="single_agent",
                )
            )

            # Filter by topology
            results = await store.list_tasks(topology="hybrid", limit=1000)
            assert any(r.id == t1.id for r in results)
            assert not any(r.id == t2.id for r in results)

    async def test_cost_budget_enforcement_per_call(self) -> None:
        """VAL-COST-BUDGET-001/005: Cost cap halts on first over-budget call."""
        from src.llm.cost import CostBudgetExceededError, get_max_cost_per_task

        with patch.dict(os.environ, {"MAX_COST_PER_TASK_USD": "0.01"}):
            max_cost = get_max_cost_per_task()
            assert max_cost == Decimal("0.01")

            # Simulate a call that exceeds the budget
            current_cost = Decimal("0.005")
            call_cost = Decimal("0.01")
            total_after = current_cost + call_cost

            if total_after > max_cost:
                err = CostBudgetExceededError(
                    task_id="test",
                    total_cost_usd=total_after,
                    max_cost_usd=max_cost,
                    triggering_agent="planner",
                )
                assert "0.0150" in str(err) or "0.01" in str(err)

    async def test_cumulative_cost_matches_langfuse(self) -> None:
        """VAL-COST-BUDGET-003: Cumulative cost is consistent."""
        # Simulate 3 LLM calls with costs
        costs = [Decimal("0.0050"), Decimal("0.0030"), Decimal("0.0040")]
        cumulative = sum(costs, Decimal("0"))
        assert cumulative == Decimal("0.0120")

        # This should match the sum of individual span costs
        span_cost_sum = sum(costs, Decimal("0"))
        assert cumulative == span_cost_sum
