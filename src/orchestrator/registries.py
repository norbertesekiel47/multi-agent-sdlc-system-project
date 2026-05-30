"""Per-task registries for sandbox, store, semantic-store, and guardrail handles.

LangGraph serializes state between nodes, so non-serializable handles
(SandboxManager, EpisodicStore, SemanticStore, GuardrailMiddleware, and the
GuardrailSandboxProxy) cannot live in the graph state.  Instead, the
Orchestrator registers these handles here keyed by ``task_id`` before the graph
runs, node functions look them up during execution, and the Orchestrator
unregisters them afterwards.
"""

from __future__ import annotations

from src.guardrails.middleware import GuardrailMiddleware, GuardrailSandboxProxy
from src.memory.episodic.store import EpisodicStore
from src.memory.semantic.store import SemanticStore
from src.sandbox.manager import SandboxManager

# ── Sandbox Registry ───────────────────────────────────────────────
# LangGraph serializes state between nodes, so SandboxManager
# objects cannot live in the state.  Instead, we use a task-scoped
# registry that nodes look up by task_id.  The Orchestrator
# provisions the sandbox before the graph runs and tears it down
# after.

_active_sandboxes: dict[str, SandboxManager] = {}


def register_sandbox(task_id: str, sandbox: SandboxManager) -> None:
    """Register a sandbox for a task (called by Orchestrator before graph)."""
    _active_sandboxes[task_id] = sandbox


def get_sandbox(task_id: str) -> SandboxManager | None:
    """Look up the sandbox for a task (called by node functions)."""
    return _active_sandboxes.get(task_id)


def unregister_sandbox(task_id: str) -> None:
    """Remove a sandbox from the registry (called by Orchestrator after graph)."""
    _active_sandboxes.pop(task_id, None)


# ── Episodic Store Registry ─────────────────────────────────────────
# Same pattern as sandbox — store reference can't be serialized.

_active_stores: dict[str, EpisodicStore] = {}
_active_semantic_stores: dict[str, SemanticStore] = {}


def register_store(task_id: str, store: EpisodicStore) -> None:
    """Register an episodic store for a task."""
    _active_stores[task_id] = store


def get_store(task_id: str) -> EpisodicStore | None:
    """Look up the episodic store for a task."""
    return _active_stores.get(task_id)


def unregister_store(task_id: str) -> None:
    """Remove an episodic store from the registry."""
    _active_stores.pop(task_id, None)


def register_semantic_store(task_id: str, store: SemanticStore) -> None:
    """Register a semantic store for a task."""
    _active_semantic_stores[task_id] = store


def get_semantic_store(task_id: str) -> SemanticStore | None:
    """Look up the semantic store for a task."""
    return _active_semantic_stores.get(task_id)


def unregister_semantic_store(task_id: str) -> SemanticStore | None:
    """Remove a semantic store from the registry and return it."""
    return _active_semantic_stores.pop(task_id, None)


# ── Guardrail Registry ────────────────────────────────────────────
# Invariant guardrails (VAL-GUARDRAIL-001 through VAL-GUARDRAIL-011).
# The GuardrailSandboxProxy wraps the SandboxManager and intercepts
# tool calls before dispatch.

_active_guardrails: dict[str, GuardrailMiddleware] = {}
_active_sandbox_proxies: dict[str, GuardrailSandboxProxy] = {}


def register_guardrail(task_id: str, guardrail: GuardrailMiddleware) -> None:
    """Register a guardrail middleware and sandbox proxy for a task.

    Creates a GuardrailSandboxProxy wrapping the task's sandbox
    so that agent tool calls go through guardrail checks.
    """
    _active_guardrails[task_id] = guardrail
    sandbox = get_sandbox(task_id)
    if sandbox is not None:
        proxy = GuardrailSandboxProxy(sandbox, guardrail)
        _active_sandbox_proxies[task_id] = proxy


def get_guardrail(task_id: str) -> GuardrailMiddleware | None:
    """Look up the guardrail middleware for a task."""
    return _active_guardrails.get(task_id)


def get_sandbox_proxy(task_id: str) -> GuardrailSandboxProxy | None:
    """Look up the guardrail-wrapped sandbox proxy for a task.

    This should be used in agent deps instead of the raw sandbox
    so that all tool calls go through guardrail checks.
    """
    return _active_sandbox_proxies.get(task_id)


def unregister_guardrail(task_id: str) -> None:
    """Remove guardrail and proxy from the registry."""
    _active_guardrails.pop(task_id, None)
    _active_sandbox_proxies.pop(task_id, None)
