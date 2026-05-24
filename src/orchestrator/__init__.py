"""LangGraph orchestrator — coordinates agent workflows.

Builds the LangGraph state machine for each topology and runs
the workflow end-to-end. The ``single_agent`` topology uses one
PydanticAI agent with all tools (VAL-TOPOLOGY-001).
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from src.agents.models import IssueContext, SingleAgentOutput
from src.agents.single_agent import SandboxTools, single_agent
from src.github_client.client import GitHubClient, canonicalize_repo_url
from src.llm.cost import estimate_cost_tiktoken, get_max_cost_per_task
from src.logging.secret_filter import install_secret_filter
from src.memory.episodic.models import (
    CreateOutcomeParams,
    UpsertRepoFactParams,
)
from src.memory.episodic.store import EpisodicStore
from src.sandbox.manager import SandboxManager
from src.tracing.langfuse import SpanType, get_tracing_client
from src.tracing.ws_broadcaster import TraceEvent, get_trace_broadcaster

load_dotenv()

# Install secret filter globally
install_secret_filter()

logger = logging.getLogger(__name__)

# Branch prefix
_BRANCH_PREFIX = "sdlc-swarm/"


# ── Graph state ─────────────────────────────────────────────────────


class OrchestratorState(BaseModel):
    """State that flows through the LangGraph nodes.

    This is the typed state that each node reads and writes.
    """

    task_id: str = ""
    repo_url: str = ""
    issue_number: int = 0
    issue_text: str = ""
    topology: str = "single_agent"
    status: str = "running"

    # Agent outputs
    issue_context: IssueContext | None = None
    agent_output: SingleAgentOutput | None = None

    # Cost tracking
    total_cost_usd: Decimal = Decimal("0")
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens_cached: int = 0

    # Error tracking
    errors: list[str] = []
    retry_count: int = 0

    # Result
    pr_url: str = ""
    outcome: str = ""

    model_config = {"arbitrary_types_allowed": True}


# ── Node functions ──────────────────────────────────────────────────


async def run_single_agent(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the single-agent topology.

    Creates a PydanticAI agent with all tools, feeds it the issue
    context, and collects the typed output.
    """
    task_id = state.task_id
    repo_url = state.repo_url
    issue_text = state.issue_text
    issue_number = state.issue_number

    logger.info(
        "Running single_agent for task %s (issue #%d)", task_id, issue_number
    )

    # Create a fresh sandbox for this run
    sandbox = SandboxManager(task_id=task_id)
    await sandbox.setup()

    # Clone the repo
    pat = os.getenv("GITHUB_PAT", "")
    username = os.getenv("GITHUB_USERNAME", "")
    gh_client = GitHubClient(pat=pat, username=username)

    try:
        gh_client.clone(repo_url, str(sandbox.workspace_dir))
    except Exception as exc:
        logger.error("Clone failed for task %s: %s", task_id, exc)
        await sandbox.teardown()
        return {"errors": [f"Clone failed: {exc}"], "outcome": "sandbox_failure"}

    # Create trace in Langfuse
    tracing = get_tracing_client()
    trace_id = str(uuid4())
    trace_span_id = tracing.create_trace(
        trace_id=trace_id,
        name=f"task.{task_id}.single_agent",
        metadata={"task_id": task_id, "topology": "single_agent"},
    )

    # Create agent span
    agent_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=trace_span_id,
        name="single_agent",
        span_type=SpanType.SPAN,
        input_data={"issue_text": issue_text[:500]},
        metadata={"task_id": task_id, "agent_name": "single_agent"},
        start_time=datetime.now(UTC),
    )

    # Broadcast agent start event
    broadcaster = get_trace_broadcaster()
    await broadcaster.publish(
        TraceEvent(
            type="node_start",
            task_id=task_id,
            trace_id=trace_id,
            span_id=agent_span_id or "",
            parent_span_id=trace_span_id,
            name="single_agent",
            span_type=SpanType.SPAN,
            metadata={"agent_name": "single_agent"},
        )
    )

    # Prepare issue context — read repo files
    repo_files: dict[str, str] = {}
    try:
        file_list = await sandbox.run_command(
            "find . -type f -name '*.py' -not -path '*/.git/*' | head -20"
        )
        for filepath in file_list.strip().split("\n"):
            filepath = filepath.strip().lstrip("./")
            if filepath and filepath.endswith(".py"):
                try:
                    content = await sandbox.read_file(filepath)
                    repo_files[filepath] = content[:2000]
                except Exception:
                    pass
    except Exception:
        pass

    issue_context = IssueContext(
        repo_url=repo_url,
        issue_number=issue_number,
        issue_text=issue_text,
        repo_files=repo_files,
    )

    # Create agent deps
    tools = SandboxTools(
        sandbox_manager=sandbox,
        workspace_dir=str(sandbox.workspace_dir),
    )

    # Run the agent
    try:
        result = await single_agent.run(
            issue_context.model_dump_json(),
            deps=tools,
        )
        agent_output = result.output

        # Accumulate cost from the agent run
        cost_usd = Decimal("0")
        tokens_in = 0
        tokens_out = 0
        cached_tokens = 0

        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            tokens_in = getattr(usage, "request_tokens", 0) or 0
            tokens_out = getattr(usage, "response_tokens", 0) or 0

        # Estimate cost based on token usage
        cost_usd = estimate_cost_tiktoken(
            model="deepseek/deepseek-v4-flash",
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
        )

        new_total = state.total_cost_usd + cost_usd

        # Check cost budget
        max_cost = get_max_cost_per_task()
        if new_total > max_cost:
            logger.warning(
                "Cost budget exceeded for task %s: $%s > $%s",
                task_id,
                new_total,
                max_cost,
            )
            await sandbox.teardown()
            return {
                "outcome": "cost_budget_exhausted",
                "total_cost_usd": new_total,
                "status": "failed",
            }

        # Update agent span end
        tracing.update_span(
            trace_id=trace_id,
            span_id=agent_span_id or "",
            output_data=agent_output.model_dump(mode="json"),
            end_time=datetime.now(UTC),
            metadata={
                "agent_name": "single_agent",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": str(cost_usd),
            },
        )

        # Broadcast agent end event
        await broadcaster.publish(
            TraceEvent(
                type="node_end",
                task_id=task_id,
                trace_id=trace_id,
                span_id=agent_span_id or "",
                parent_span_id=trace_span_id,
                name="single_agent",
                span_type=SpanType.SPAN,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cached_tokens=cached_tokens,
                cost_usd=cost_usd,
                metadata={"agent_name": "single_agent"},
            )
        )

        logger.info(
            "Agent completed for task %s: ready_for_pr=%s, cost=$%s",
            task_id,
            agent_output.ready_for_pr,
            cost_usd,
        )

        return {
            "agent_output": agent_output,
            "total_cost_usd": new_total,
            "total_tokens_in": state.total_tokens_in + tokens_in,
            "total_tokens_out": state.total_tokens_out + tokens_out,
            "total_tokens_cached": state.total_tokens_cached + cached_tokens,
            "status": "completed" if agent_output.ready_for_pr else "running",
        }

    except Exception as exc:
        logger.error("Agent failed for task %s: %s", task_id, exc)
        tracing.update_span(
            trace_id=trace_id,
            span_id=agent_span_id or "",
            output_data={"error": str(exc)[:500]},
            end_time=datetime.now(UTC),
            level="ERROR",
        )
        await sandbox.teardown()
        return {"errors": [f"Agent failed: {exc}"], "outcome": "failed"}

    finally:
        with contextlib.suppress(Exception):
            await sandbox.teardown()

        # Flush tracing
        tracing.flush()


async def commit_and_pr(state: OrchestratorState) -> dict[str, Any]:
    """Node: Commit changes and open a PR.

    Only runs if the agent output has ready_for_pr=True.
    """
    task_id = state.task_id
    repo_url = state.repo_url

    if state.agent_output is None or not state.agent_output.ready_for_pr:
        logger.info("Task %s not ready for PR, skipping", task_id)
        return {"outcome": "completed", "status": "completed"}

    logger.info("Opening PR for task %s", task_id)

    pat = os.getenv("GITHUB_PAT", "")
    username = os.getenv("GITHUB_USERNAME", "")
    gh_client = GitHubClient(pat=pat, username=username)

    repo_slug = canonicalize_repo_url(repo_url).replace(
        "https://github.com/", ""
    )

    # Clone the repo fresh for commit+push
    sandbox = SandboxManager(task_id=task_id)
    await sandbox.setup()

    try:
        gh_client.clone(repo_url, str(sandbox.workspace_dir))

        # Create branch
        branch_name = f"{_BRANCH_PREFIX}fix-issue-{state.issue_number}"
        gh_client.create_branch(str(sandbox.workspace_dir), branch_name)

        # Apply the agent's code edit
        if state.agent_output.code_edit.diff:
            with contextlib.suppress(Exception):
                await sandbox.apply_diff(state.agent_output.code_edit.diff)

        # Commit and push
        summary = state.agent_output.summary[:200]
        commit_msg = (
            f"fix: resolve issue #{state.issue_number}\n\n{summary}"
        )
        gh_client.commit_and_push(
            str(sandbox.workspace_dir), branch_name, commit_msg
        )

        # Open PR
        pr_title = f"fix: resolve issue #{state.issue_number}"
        pr_body = (
            state.agent_output.summary
            or f"Automated fix for issue #{state.issue_number}"
        )
        pr_ref = gh_client.open_pull_request(
            repo=repo_slug,
            head_branch=branch_name,
            title=pr_title,
            body=pr_body,
        )

        logger.info("PR opened: %s", pr_ref.html_url)
        return {
            "pr_url": pr_ref.html_url,
            "outcome": "pr_opened",
            "status": "completed",
        }

    except Exception as exc:
        logger.error("PR failed for task %s: %s", task_id, exc)
        return {"errors": [f"PR failed: {exc}"], "outcome": "failed"}

    finally:
        with contextlib.suppress(Exception):
            await sandbox.teardown()


# ── Build the single_agent graph ────────────────────────────────────


def build_single_agent_graph() -> StateGraph:
    """Build the LangGraph state machine for the single_agent topology.

    The graph is linear: START → run_single_agent → commit_and_pr → END

    The Langfuse trace will contain exactly one distinct agent name
    "single_agent" (VAL-TOPOLOGY-001).
    """
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("run_single_agent", run_single_agent)
    graph.add_node("commit_and_pr", commit_and_pr)

    # Add edges
    graph.add_edge(START, "run_single_agent")
    graph.add_edge("run_single_agent", "commit_and_pr")
    graph.add_edge("commit_and_pr", END)

    return graph


# ── Orchestrator ────────────────────────────────────────────────────


class Orchestrator:
    """High-level orchestrator that runs a task end-to-end.

    Usage::

        orchestrator = Orchestrator(store=episodic_store)
        result = await orchestrator.run_task(task_id=uuid)
    """

    def __init__(self, store: EpisodicStore) -> None:
        self.store = store

    async def run_task(self, task_id: str) -> OrchestratorState:
        """Run a task through the configured topology.

        Steps:
        1. Load the task from the episodic store
        2. Build the appropriate LangGraph topology
        3. Execute the graph
        4. Persist outcomes and update task status
        5. Return the final state
        """
        task = await self.store.get_task(UUID(task_id))
        if task is None:
            msg = f"Task {task_id} not found"
            raise ValueError(msg)

        logger.info(
            "Starting task %s (topology=%s, repo=%s, issue=%d)",
            task_id,
            task.topology,
            task.repo_url,
            task.issue_number,
        )

        # Build initial state
        initial_state = OrchestratorState(
            task_id=task_id,
            repo_url=task.repo_url,
            issue_number=task.issue_number or 0,
            issue_text=task.issue_text,
            topology=task.topology,
            status="running",
        )

        # Build and compile the graph
        if task.topology == "single_agent":
            graph = build_single_agent_graph()
        else:
            msg = (
                f"Topology {task.topology!r} not yet implemented "
                "(M1 only supports single_agent)"
            )
            raise ValueError(msg)

        compiled = graph.compile()

        # Execute the graph
        try:
            result = await compiled.ainvoke(initial_state)
        except Exception as exc:
            logger.error(
                "Graph execution failed for task %s: %s", task_id, exc
            )
            await self.store.finish_task(UUID(task_id), "failed")
            await self.store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="sandbox_failure",
                    detail={"error": str(exc)[:500]},
                )
            )
            return initial_state.model_copy(
                update={"status": "failed", "outcome": "sandbox_failure"}
            )

        # Process the result
        final_state = OrchestratorState.model_validate(result)

        # Update the task in the episodic store
        await self.store.update_task_totals(
            UUID(task_id),
            total_cost_usd=final_state.total_cost_usd,
            total_tokens_in=final_state.total_tokens_in,
            total_tokens_out=final_state.total_tokens_out,
            total_tokens_cached=final_state.total_tokens_cached,
        )

        # Write outcome
        if final_state.outcome:
            await self.store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome=final_state.outcome,
                    detail={
                        "pr_url": final_state.pr_url,
                        "errors": final_state.errors,
                    },
                )
            )

        # Finish the task
        await self.store.finish_task(
            UUID(task_id),
            final_state.status,
            pr_url=final_state.pr_url or None,
        )

        # Write a repo_fact for this repo
        await self.store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=task.repo_url,
                fact_kind="last_topology_success",
                fact_value={
                    "topology": task.topology,
                    "outcome": final_state.outcome,
                    "cost_usd": str(final_state.total_cost_usd),
                },
            )
        )

        logger.info(
            "Task %s completed: status=%s, outcome=%s, pr_url=%s, cost=$%s",
            task_id,
            final_state.status,
            final_state.outcome,
            final_state.pr_url,
            final_state.total_cost_usd,
        )

        return final_state
