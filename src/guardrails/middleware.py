"""Guardrail middleware that intercepts tool calls before dispatch.

VAL-GUARDRAIL-011: The middleware sits between the orchestrator/agents
and the sandbox executor.  When a rule fires, the sandbox executor
is NOT called.

The middleware wraps the SandboxManager's tool surface (run_command,
write_file, read_file, apply_diff, run_tests) and checks every call
against the registered guardrail rules before delegating.

When a violation is detected:
  1. A ``GuardrailViolation`` is raised (VAL-GUARDRAIL-007, VAL-GUARDRAIL-008, VAL-GUARDRAIL-009)
  2. The caller (orchestrator node) catches it and:
     - Logs to Langfuse (span tagged ``guardrail.violation``)
     - Writes an ``outcomes`` row with ``outcome='guardrail_block'``
     - Halts the agent step
     - Escalates to HITL via ``interrupt()``
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.guardrails.errors import GuardrailViolation
from src.guardrails.rules import (
    GuardrailRuleBase,
    SecretLeakRule,
    create_default_rules,
)
from src.sandbox.config import SANDBOX_REPO_MOUNT_POINT
from src.sandbox.manager import SandboxManager

logger = logging.getLogger(__name__)


def _load_secret_values() -> list[str]:
    """Load secret values from the environment.

    These are the actual values (not variable names) that must never
    appear in subprocess commands (VAL-GUARDRAIL-005).
    """
    secret_env_keys = [
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_PAT",
        "HUGGINGFACE_TOKEN",
    ]
    values: list[str] = []
    for key in secret_env_keys:
        val = os.getenv(key, "")
        if val:
            values.append(val)
    return values


class GuardrailMiddleware:
    """Middleware that intercepts sandbox tool calls and checks guardrail rules.

    Usage::

        middleware = GuardrailMiddleware(sandbox_cwd="/workspace")
        sandbox = SandboxManager(task_id="abc")

        # Checked call — raises GuardrailViolation if blocked
        await middleware.run_command(sandbox=sandbox, command="pip install requests")

        # Direct call — bypasses guardrails (NOT recommended)
        await sandbox.run_command("pip install requests")

    The middleware wraps all five sandbox tool methods:
    ``run_command``, ``write_file``, ``read_file``, ``apply_diff``, ``run_tests``.
    """

    def __init__(
        self,
        *,
        sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
        secret_values: list[str] | None = None,
        allowed_hosts: list[str] | None = None,
        rules: list[GuardrailRuleBase] | None = None,
    ) -> None:
        self.sandbox_cwd = sandbox_cwd.rstrip("/")

        if rules is not None:
            self._rules: list[GuardrailRuleBase] = rules
        else:
            # Load secret values from env if not explicitly provided
            secrets = secret_values if secret_values is not None else _load_secret_values()
            self._rules = create_default_rules(
                secret_values=secrets,
                allowed_hosts=allowed_hosts,
            )

    @property
    def rules(self) -> list[GuardrailRuleBase]:
        """Return the list of active guardrail rules."""
        return list(self._rules)

    def check_command(self, command: str) -> GuardrailViolation | None:
        """Check a command against all guardrail rules.

        Returns the first violation found, or None if the command is allowed.
        This is a synchronous check (no sandbox I/O needed).
        """
        for rule in self._rules:
            violation = rule.check(
                tool_name="run_command",
                command=command,
                sandbox_cwd=self.sandbox_cwd,
            )
            if violation is not None:
                return violation
        return None

    def check_content(
        self,
        content: str,
        tool_name: str = "write_file",
    ) -> GuardrailViolation | None:
        """Check file content against relevant guardrail rules (secret leak).

        Only the SecretLeakRule applies to file content.
        """
        for rule in self._rules:
            if isinstance(rule, SecretLeakRule):
                violation = rule.check(
                    tool_name=tool_name,
                    command=content,
                )
                if violation is not None:
                    return violation
        return None

    # ── Sandbox tool wrappers ──────────────────────────────────────

    async def run_command(
        self,
        sandbox: SandboxManager,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Execute a command inside the sandbox after guardrail checks.

        Raises GuardrailViolation if any rule fires.
        """
        violation = self.check_command(command)
        if violation is not None:
            logger.warning(
                "Guardrail blocked command: rule=%s, cmd=%.100s",
                violation.rule_name,
                command,
            )
            raise violation

        return await sandbox.run_command(command, cwd=cwd, timeout=timeout)

    async def write_file(
        self,
        sandbox: SandboxManager,
        path: str,
        content: str,
    ) -> None:
        """Write a file inside the sandbox after guardrail checks.

        Checks file content for secret leaks.
        """
        violation = self.check_content(content, tool_name="write_file")
        if violation is not None:
            logger.warning(
                "Guardrail blocked write_file: rule=%s, path=%s",
                violation.rule_name,
                path,
            )
            raise violation

        await sandbox.write_file(path, content)

    async def read_file(
        self,
        sandbox: SandboxManager,
        path: str,
    ) -> str:
        """Read a file from the sandbox (no guardrail check needed for reads)."""
        return await sandbox.read_file(path)

    async def apply_diff(
        self,
        sandbox: SandboxManager,
        diff: str,
    ) -> None:
        """Apply a diff inside the sandbox after guardrail checks.

        Checks diff content for secret leaks.
        """
        violation = self.check_content(diff, tool_name="apply_diff")
        if violation is not None:
            logger.warning(
                "Guardrail blocked apply_diff: rule=%s",
                violation.rule_name,
            )
            raise violation

        await sandbox.apply_diff(diff)

    async def run_tests(
        self,
        sandbox: SandboxManager,
        test_command: str = "pytest",
    ) -> str:
        """Run tests inside the sandbox after guardrail checks."""
        violation = self.check_command(test_command)
        if violation is not None:
            logger.warning(
                "Guardrail blocked run_tests: rule=%s, cmd=%.100s",
                violation.rule_name,
                test_command,
            )
            raise violation

        return await sandbox.run_tests(test_command)


# ── Helper: report a violation to Langfuse + episodic store ────────


async def report_guardrail_violation(
    *,
    violation: GuardrailViolation,
    task_id: str,
    trace_id: str,
    parent_span_id: str | None = None,
    sandbox_cwd: str = SANDBOX_REPO_MOUNT_POINT,
) -> None:
    """Report a guardrail violation to Langfuse and write an outcomes row.

    VAL-GUARDRAIL-007: Emits a Langfuse span tagged ``guardrail.violation``.
    VAL-GUARDRAIL-008: Writes an ``outcomes`` row with ``outcome='guardrail_block'``.
    """
    from datetime import UTC, datetime

    from src.tracing.langfuse import SpanType, get_tracing_client
    from src.tracing.ws_broadcaster import TraceEvent, get_trace_broadcaster

    # ── Langfuse span ──────────────────────────────────────────────
    tracing = get_tracing_client()
    report = violation.to_langfuse_report()

    span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name="guardrail.violation",
        span_type=SpanType.SPAN,
        input_data={"rule_name": violation.rule_name, "tool_name": violation.tool_name},
        output_data=report,
        metadata={
            "guardrail_rule": violation.rule_name,
            "tool_name": violation.tool_name,
            "detail": violation.detail,
        },
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        level="ERROR",
        status_message=f"Guardrail violation: {violation.rule_name}",
    )

    # ── WebSocket broadcast ────────────────────────────────────────
    broadcaster = get_trace_broadcaster()
    await broadcaster.publish(
        TraceEvent(
            type="guardrail_violation",
            task_id=task_id,
            trace_id=trace_id,
            span_id=span_id or "",
            parent_span_id=parent_span_id or "",
            name="guardrail.violation",
            span_type=SpanType.SPAN,
            metadata=report,
        )
    )

    # ── Outcomes row ───────────────────────────────────────────────
    from uuid import UUID

    from src.memory.episodic.models import CreateOutcomeParams
    from src.memory.episodic.store import EpisodicStore

    try:
        store = EpisodicStore()
        await store.connect()
        try:
            outcome_data = violation.to_outcome_data()
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome=outcome_data["outcome"],
                    detail=outcome_data["detail"],
                )
            )
        finally:
            await store.close()
    except Exception:
        logger.error(
            "Failed to write guardrail_block outcome for task %s",
            task_id,
            exc_info=True,
        )

    logger.info(
        "Guardrail violation reported: task=%s, rule=%s, tool=%s",
        task_id,
        violation.rule_name,
        violation.tool_name,
    )


# ── GuardrailSandboxProxy ────────────────────────────────────────────


class GuardrailSandboxProxy:
    """Drop-in replacement for SandboxManager in agent deps.

    Wraps a SandboxManager with guardrail checks.  Agents receive
    this proxy as their ``sandbox_manager`` dep — every tool call
    goes through the guardrail middleware before reaching the
    underlying sandbox (VAL-GUARDRAIL-011).

    Usage::

        sandbox = SandboxManager(task_id="abc")
        guardrail = GuardrailMiddleware()
        proxy = GuardrailSandboxProxy(sandbox, guardrail)

        # Agents use proxy just like a SandboxManager
        await proxy.run_command("pip install requests")  # checked

    Non-tool methods (``is_running``, ``container_id``, ``teardown``)
    delegate directly to the underlying sandbox without guardrail
    checks.
    """

    def __init__(
        self,
        sandbox: SandboxManager,
        guardrail: GuardrailMiddleware,
    ) -> None:
        self._sandbox = sandbox
        self._guardrail = guardrail

    # ── Tool surface (with guardrail checks) ──────────────────────

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Execute a command after guardrail check."""
        return await self._guardrail.run_command(
            self._sandbox, command, cwd=cwd, timeout=timeout
        )

    async def write_file(self, path: str, content: str) -> None:
        """Write a file after guardrail check."""
        await self._guardrail.write_file(self._sandbox, path, content)

    async def read_file(self, path: str) -> str:
        """Read a file (no guardrail check for reads)."""
        return await self._guardrail.read_file(self._sandbox, path)

    async def apply_diff(self, diff: str) -> None:
        """Apply a diff after guardrail check."""
        await self._guardrail.apply_diff(self._sandbox, diff)

    async def run_tests(self, test_command: str = "pytest") -> str:
        """Run tests after guardrail check."""
        return await self._guardrail.run_tests(
            self._sandbox, test_command
        )

    # ── Direct delegation (no guardrail check) ─────────────────────

    @property
    def is_running(self) -> bool:
        """Check if the sandbox container is running."""
        return self._sandbox.is_running

    @property
    def container_id(self) -> str | None:
        """Return the sandbox container ID."""
        return self._sandbox.container_id

    @property
    def task_id(self) -> str:
        """Return the task ID."""
        return self._sandbox.task_id

    @property
    def workspace_dir(self) -> Any:
        """Return the workspace directory path."""
        return self._sandbox.workspace_dir

    async def setup(self) -> None:
        """Provision the sandbox (delegates directly)."""
        await self._sandbox.setup()

    async def teardown(self) -> None:
        """Tear down the sandbox (delegates directly)."""
        await self._sandbox.teardown()

    # ── Expose underlying sandbox for direct access ───────────────

    @property
    def unwrapped(self) -> SandboxManager:
        """Return the underlying SandboxManager (bypasses guardrails)."""
        return self._sandbox
