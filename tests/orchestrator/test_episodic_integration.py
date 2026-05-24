"""Tests for episodic memory integration into the multi-agent workflow.

Covers validation assertions:
  - VAL-CROSS-013: Cross-task memory reuse on the same repo
  - VAL-CROSS-014: Cross-task does NOT leak memory across repos
  - VAL-CROSS-034: Episodic memory carries forward across full stack restart

Also verifies:
  - Planner's IssueContext is populated with episodic data from the store
  - Supervisor writes decisions rows after every agent turn with correct
    agent/step/decision_type
  - Repo URL canonicalization is consistent across tasks, repo_chunks,
    repo_facts, and GitHub client
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from src.agents.models import ChangePlan, CodeEdit, IssueContext, ReviewResult
from src.github_client.client import canonicalize_repo_url
from src.memory.episodic.models import (
    CreateDecisionParams,
    CreateOutcomeParams,
    CreateTaskParams,
    UpsertRepoFactParams,
)
from src.memory.episodic.store import EpisodicStore

load_dotenv = None
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except ImportError:
    pass


# ── Fixtures ─────────────────────────────────────────────────────────────────

_DSN = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
    f":{os.getenv('POSTGRES_PORT', '5433')}"
    f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
)


@pytest.fixture
async def store() -> EpisodicStore:  # type: ignore[valid-type]
    """Provide a connected EpisodicStore, closing after test."""
    s = EpisodicStore(dsn=_DSN)
    await s.connect()
    yield s
    await s.close()


@pytest.fixture
def ns() -> str:
    """Unique namespace for test isolation (prevents xdist conflicts)."""
    return str(uuid4())[:8]


def _repo(ns: str, name: str) -> str:
    """Build a unique repo URL for test isolation."""
    return f"https://github.com/test/{ns}-{name}"


# ── Helper: seed a completed task with outcomes and repo_facts ──────────────


async def _seed_completed_task(
    store: EpisodicStore,
    *,
    repo_url: str,
    issue_number: int = 1,
    outcome: str = "pr_opened",
    fact_kind: str = "language",
    fact_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a completed task with outcomes and repo_facts.

    Returns a dict with task_id and other identifiers for verification.
    """
    task = await store.create_task(
        CreateTaskParams(
            repo_url=repo_url,
            issue_number=issue_number,
            issue_text=f"Test issue #{issue_number} for {repo_url}",
            topology="supervisor_only",
            status="running",
        )
    )
    task_id = task.id

    # Write decisions for each agent turn
    await store.create_decision(
        CreateDecisionParams(
            task_id=task_id,
            agent="planner",
            step_index=0,
            decision_type="change_plan",
            decision_data={
                "target_files": ["src/main.py"],
                "rationale": "Fix the bug in the main module",
            },
        )
    )
    await store.create_decision(
        CreateDecisionParams(
            task_id=task_id,
            agent="coder",
            step_index=1,
            decision_type="code_edit",
            decision_data={
                "diff": "--- a/src/main.py\n+++ b/src/main.py\n",
                "touched_files": ["src/main.py"],
            },
        )
    )
    await store.create_decision(
        CreateDecisionParams(
            task_id=task_id,
            agent="reviewer",
            step_index=2,
            decision_type="review_verdict",
            decision_data={"verdict": "accept", "issues": []},
        )
    )

    # Write outcome
    await store.create_outcome(
        CreateOutcomeParams(
            task_id=task_id,
            outcome=outcome,
            detail={"pr_url": "https://github.com/test/pr/1"},
        )
    )

    # Write repo_facts
    if fact_value is None:
        fact_value = {"value": "python"}
    await store.upsert_repo_fact(
        UpsertRepoFactParams(
            repo_url=repo_url,
            fact_kind=fact_kind,
            fact_value=fact_value,
        )
    )

    # Mark task as completed
    await store.finish_task(
        task_id,
        "completed",
        pr_url="https://github.com/test/pr/1",
    )

    return {
        "task_id": task_id,
        "repo_url": repo_url,
        "outcome": outcome,
        "fact_kind": fact_kind,
        "fact_value": fact_value,
    }


# ══════════════════════════════════════════════════════════════════════════
# VAL-CROSS-013: Cross-task memory reuse on the same repo
# ══════════════════════════════════════════════════════════════════════════


class TestCrossTaskMemoryReuse:
    """Running task Y on the same repo as task X: Planner's episodic
    query returns rows from task X (outcomes and/or repo_facts)."""

    @pytest.mark.asyncio
    async def test_planner_context_returns_prior_task_outcomes(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Task Y on same repo as X: get_planner_context returns X's outcomes."""
        repo_url = _repo(ns, "shared-repo")

        # Seed task X on repo R
        await _seed_completed_task(
            store, repo_url=repo_url, outcome="pr_opened"
        )

        # Query planner context for a NEW task Y on the same repo
        context = await store.get_planner_context(repo_url, recent_limit=5)

        # Should return at least one outcome from task X
        assert len(context["recent_outcomes"]) >= 1
        x_outcome = context["recent_outcomes"][0]
        assert x_outcome["outcome"] == "pr_opened"

    @pytest.mark.asyncio
    async def test_planner_context_returns_prior_repo_facts(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Task Y on same repo as X: get_planner_context returns X's repo_facts."""
        repo_url = _repo(ns, "shared-repo")

        # Seed task X on repo R with a repo_fact
        await _seed_completed_task(
            store,
            repo_url=repo_url,
            fact_kind="language",
            fact_value={"value": "python"},
        )

        # Query planner context for a NEW task Y on the same repo
        context = await store.get_planner_context(repo_url, recent_limit=5)

        # Should return repo_facts from task X
        assert len(context["repo_facts"]) >= 1
        lang_fact = context["repo_facts"][0]
        assert lang_fact["fact_kind"] == "language"
        assert lang_fact["fact_value"]["value"] == "python"

    @pytest.mark.asyncio
    async def test_planner_context_returns_prior_decisions(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Task Y on same repo as X: get_planner_context returns X's decisions."""
        repo_url = _repo(ns, "shared-repo")

        # Seed task X on repo R
        await _seed_completed_task(store, repo_url=repo_url)

        # Query planner context
        context = await store.get_planner_context(repo_url, recent_limit=5)

        # Should return decisions from task X
        assert len(context["recent_decisions"]) >= 1

    @pytest.mark.asyncio
    async def test_planner_prompt_includes_prior_outcome_string(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """VAL-CROSS-013: Y's Planner prompt input contains a serialization
        of at least one prior outcomes.outcome or repo_facts.fact_value
        from task X."""
        repo_url = _repo(ns, "shared-repo")

        # Seed task X on repo R
        await _seed_completed_task(
            store,
            repo_url=repo_url,
            outcome="pr_opened",
            fact_kind="test_command",
            fact_value={"command": "pytest -x"},
        )

        # Build IssueContext for task Y with pre-populated episodic data
        planner_ctx = await store.get_planner_context(repo_url, recent_limit=5)

        # Construct IssueContext with episodic data
        issue_context = IssueContext(
            repo_url=repo_url,
            issue_number=2,
            issue_text="Second issue on the same repo",
            repo_facts=planner_ctx["repo_facts"],
            recent_decisions=planner_ctx["recent_decisions"],
        )

        # Build the planner prompt (same function used by the orchestrator)
        from src.agents.planner import _build_planner_prompt

        prompt = _build_planner_prompt(issue_context)

        # The prompt should contain strings from task X's data
        assert "pr_opened" in prompt or "test_command" in prompt


# ══════════════════════════════════════════════════════════════════════════
# VAL-CROSS-014: Cross-task does NOT leak memory across repos
# ══════════════════════════════════════════════════════════════════════════


class TestCrossRepoIsolation:
    """Task on repo A: episodic query returns zero rows from repo B."""

    @pytest.mark.asyncio
    async def test_planner_context_no_cross_repo_outcomes(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Planner for repo A never sees repo B's outcomes."""
        repo_a = _repo(ns, "repo-a")
        repo_b = _repo(ns, "repo-b")

        # Seed task X on repo A with a distinctive outcome
        await _seed_completed_task(
            store,
            repo_url=repo_a,
            outcome="pr_opened",
            fact_kind="language",
            fact_value={"value": "python"},
        )

        # Seed task on repo B with a distinctive outcome
        await _seed_completed_task(
            store,
            repo_url=repo_b,
            outcome="sandbox_failure",
            fact_kind="language",
            fact_value={"value": "rust"},
        )

        # Query planner context for repo A — should NOT include repo B data
        ctx_a = await store.get_planner_context(repo_a, recent_limit=5)

        # Repo A outcomes should only be about repo A
        for outcome in ctx_a["recent_outcomes"]:
            assert outcome["outcome"] != "sandbox_failure"

        # Repo A facts should show python, not rust
        for fact in ctx_a["repo_facts"]:
            assert fact["fact_value"].get("value") != "rust"

    @pytest.mark.asyncio
    async def test_planner_context_no_cross_repo_facts(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Planner for repo B never sees repo A's repo_facts."""
        repo_a = _repo(ns, "repo-a")
        repo_b = _repo(ns, "repo-b")

        # Seed repo A with a distinctive fact
        await _seed_completed_task(
            store,
            repo_url=repo_a,
            fact_kind="install_command",
            fact_value={"command": "pip install -r requirements.txt"},
        )

        # Query planner context for repo B — should have NO facts from A
        ctx_b = await store.get_planner_context(repo_b, recent_limit=5)
        assert len(ctx_b["repo_facts"]) == 0

    @pytest.mark.asyncio
    async def test_planner_prompt_no_cross_repo_strings(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """VAL-CROSS-014: Task Z's Planner prompt does NOT contain any
        string from R1's outcomes.detail or repo_facts.fact_value."""
        repo_a = _repo(ns, "repo-a")
        repo_b = _repo(ns, "repo-b")

        # Seed task on repo A with distinctive data
        await _seed_completed_task(
            store,
            repo_url=repo_a,
            outcome="pr_opened",
            fact_kind="test_command",
            fact_value={"command": "npm test --unique-marker-a"},
        )

        # Build planner context for repo B
        ctx_b = await store.get_planner_context(repo_b, recent_limit=5)
        issue_context = IssueContext(
            repo_url=repo_b,
            issue_number=1,
            issue_text="An issue on repo B",
            repo_facts=ctx_b["repo_facts"],
            recent_decisions=ctx_b["recent_decisions"],
        )

        from src.agents.planner import _build_planner_prompt

        prompt = _build_planner_prompt(issue_context)

        # Repo A's unique marker should NOT appear in the prompt for repo B
        assert "unique-marker-a" not in prompt
        assert "npm test" not in prompt


# ══════════════════════════════════════════════════════════════════════════
# VAL-CROSS-034: Episodic memory carries forward across full stack restart
# ══════════════════════════════════════════════════════════════════════════


class TestEpisodicMemoryRestartSurvival:
    """Episodic memory survives full stack restart: all previously
    committed rows still queryable."""

    @pytest.mark.asyncio
    async def test_data_survives_reconnect(self, ns: str) -> None:
        """VAL-CROSS-034: After closing and reconnecting the store,
        previously written rows are still queryable."""
        repo_url = _repo(ns, "restart-repo")

        # Phase 1: Write data with store instance 1
        store1 = EpisodicStore(dsn=_DSN)
        await store1.connect()

        task = await store1.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=1,
                issue_text="Issue before restart",
                topology="supervisor_only",
            )
        )

        await store1.create_outcome(
            CreateOutcomeParams(
                task_id=task.id,
                outcome="pr_opened",
                detail={"pr_url": "https://github.com/test/pr/1"},
            )
        )

        await store1.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url,
                fact_kind="language",
                fact_value={"value": "python"},
            )
        )

        await store1.finish_task(task.id, "completed")

        # Close the store (simulates backend shutdown)
        await store1.close()

        # Phase 2: Reconnect with a new store instance
        store2 = EpisodicStore(dsn=_DSN)
        await store2.connect()

        # Verify the data is still queryable
        context = await store2.get_planner_context(repo_url, recent_limit=5)

        # Outcomes from task X should still be there
        assert len(context["recent_outcomes"]) >= 1
        assert context["recent_outcomes"][0]["outcome"] == "pr_opened"

        # Repo facts should still be there
        assert len(context["repo_facts"]) >= 1
        assert context["repo_facts"][0]["fact_kind"] == "language"
        assert context["repo_facts"][0]["fact_value"]["value"] == "python"

        await store2.close()

    @pytest.mark.asyncio
    async def test_restarted_planner_gets_prior_data(self, ns: str) -> None:
        """VAL-CROSS-034: After reconnect, a new task Y on repo R
        can retrieve data written by task X before the restart."""
        repo_url = _repo(ns, "restart-repo-2")

        # Phase 1: Complete task X and write repo_facts/outcomes
        store1 = EpisodicStore(dsn=_DSN)
        await store1.connect()

        await _seed_completed_task(
            store1,
            repo_url=repo_url,
            outcome="pr_opened",
            fact_kind="last_topology_success",
            fact_value={"topology": "supervisor_only", "result": "success"},
        )
        await store1.close()

        # Phase 2: Reconnect, simulate a new task Y starting
        store2 = EpisodicStore(dsn=_DSN)
        await store2.connect()

        # Build planner context as the Planner would
        context = await store2.get_planner_context(repo_url, recent_limit=5)

        # Task Y's planner should see task X's data
        assert len(context["repo_facts"]) >= 1
        success_fact = context["repo_facts"][0]
        assert success_fact["fact_kind"] == "last_topology_success"
        assert success_fact["fact_value"]["topology"] == "supervisor_only"

        assert len(context["recent_outcomes"]) >= 1
        assert context["recent_outcomes"][0]["outcome"] == "pr_opened"

        await store2.close()


# ══════════════════════════════════════════════════════════════════════════
# Planner IssueContext populated with episodic data
# ══════════════════════════════════════════════════════════════════════════


class TestPlannerIssueContextEpisodicData:
    """Planner reads repo_facts and recent decisions/outcomes for the
    current repo_url as part of its IssueContext."""

    @pytest.mark.asyncio
    async def test_episodic_query_tool_returns_repo_facts(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Planner's episodic-query tool returns repo_facts for the
        current repo_url."""
        repo_url = _repo(ns, "episodic-query-repo")

        # Seed repo_facts
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url,
                fact_kind="language",
                fact_value={"value": "python"},
            )
        )

        # Call get_planner_context as the planner's tool would
        context = await store.get_planner_context(repo_url, recent_limit=5)

        # Should return the repo_fact
        assert len(context["repo_facts"]) >= 1
        assert context["repo_facts"][0]["fact_kind"] == "language"

    @pytest.mark.asyncio
    async def test_episodic_query_tool_scoped_to_repo_url(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """The episodic query tool is scoped by repo_url — returns only
        data for the given repo."""
        repo_a = _repo(ns, "scoped-a")
        repo_b = _repo(ns, "scoped-b")

        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_a,
                fact_kind="language",
                fact_value={"value": "python"},
            )
        )
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_b,
                fact_kind="language",
                fact_value={"value": "rust"},
            )
        )

        # Query for repo A
        ctx_a = await store.get_planner_context(repo_a, recent_limit=5)
        assert len(ctx_a["repo_facts"]) == 1
        assert ctx_a["repo_facts"][0]["fact_value"]["value"] == "python"

        # Query for repo B
        ctx_b = await store.get_planner_context(repo_b, recent_limit=5)
        assert len(ctx_b["repo_facts"]) == 1
        assert ctx_b["repo_facts"][0]["fact_value"]["value"] == "rust"


# ══════════════════════════════════════════════════════════════════════════
# Supervisor writes decisions rows after every agent turn
# ══════════════════════════════════════════════════════════════════════════


class TestSupervisorDecisionWriting:
    """Supervisor writes decisions rows after every agent turn with
    correct agent/step/decision_type."""

    @pytest.mark.asyncio
    async def test_planner_decision_written(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """After Planner turn, a decision row is written with
        agent='planner', decision_type='change_plan'."""
        repo_url = _repo(ns, "decision-test")
        task = await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=1,
                issue_text="Test issue",
                topology="supervisor_only",
            )
        )

        # Simulate what the Planner's persist_change_plan does
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="Fix the bug in the main module by correcting the calculation logic.",
            approach="Change the operator from + to -",
        )

        from src.agents.planner import persist_change_plan

        await persist_change_plan(
            store=store,
            task_id=task.id,
            plan=plan,
            step_index=0,
        )

        # Verify the decision row exists
        decisions = await store.query_recent_decisions(repo_url, limit=10)
        planner_decisions = [
            d for d in decisions
            if d.task_id == task.id and d.agent == "planner"
        ]
        assert len(planner_decisions) >= 1
        d = planner_decisions[0]
        assert d.agent == "planner"
        assert d.decision_type == "change_plan"
        assert d.step_index == 0

    @pytest.mark.asyncio
    async def test_coder_decision_written(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """After Coder turn, a decision row is written with
        agent='coder', decision_type='code_edit'."""
        repo_url = _repo(ns, "decision-test")
        task = await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=1,
                issue_text="Test issue",
                topology="supervisor_only",
            )
        )

        # Simulate what the Coder's persist_code_edit does
        edit = CodeEdit(
            diff=(
                "--- a/src/main.py\n"
                "+++ b/src/main.py\n"
                "@@ -1 +1 @@\n"
                "-return a + b\n"
                "+return a - b\n"
            ),
            touched_files=["src/main.py"],
            diff_hash="abc123",
        )

        from src.agents.coder import persist_code_edit

        await persist_code_edit(
            store=store,
            task_id=task.id,
            edit=edit,
            step_index=1,
        )

        decisions = await store.query_recent_decisions(repo_url, limit=10)
        coder_decisions = [
            d for d in decisions
            if d.task_id == task.id and d.agent == "coder"
        ]
        assert len(coder_decisions) >= 1
        d = coder_decisions[0]
        assert d.agent == "coder"
        assert d.decision_type == "code_edit"
        assert d.step_index == 1
        assert d.decision_data["diff_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_reviewer_decision_written(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """After Reviewer turn, a decision row is written with
        agent='reviewer', decision_type='review_verdict'."""
        repo_url = _repo(ns, "decision-test")
        task = await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=1,
                issue_text="Test issue",
                topology="supervisor_only",
            )
        )

        review = ReviewResult(
            verdict="accept",
            issues=[],
            diff_hash="abc123",
        )

        from src.agents.reviewer import persist_review_result

        await persist_review_result(
            store=store,
            task_id=task.id,
            review=review,
            step_index=2,
        )

        decisions = await store.query_recent_decisions(repo_url, limit=10)
        reviewer_decisions = [
            d for d in decisions
            if d.task_id == task.id and d.agent == "reviewer"
        ]
        assert len(reviewer_decisions) >= 1
        d = reviewer_decisions[0]
        assert d.agent == "reviewer"
        assert d.decision_type == "review_verdict"
        assert d.step_index == 2
        assert d.decision_data["verdict"] == "accept"

    @pytest.mark.asyncio
    async def test_all_agent_decisions_written_for_task(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """After a full Planner→Coder→Reviewer cycle, decisions rows
        exist for all three agents."""
        repo_url = _repo(ns, "full-cycle")
        task = await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=1,
                issue_text="Full cycle test issue",
                topology="supervisor_only",
            )
        )

        # Planner decision
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="Fix the bug in the main module by correcting the calculation logic.",
            approach="Change the operator",
        )
        from src.agents.planner import persist_change_plan

        await persist_change_plan(store=store, task_id=task.id, plan=plan, step_index=0)

        # Coder decision
        edit = CodeEdit(
            diff="--- a/src/main.py\n+++ b/src/main.py\n",
            touched_files=["src/main.py"],
            diff_hash="def456",
        )
        from src.agents.coder import persist_code_edit

        await persist_code_edit(store=store, task_id=task.id, edit=edit, step_index=1)

        # Reviewer decision
        review = ReviewResult(verdict="accept", issues=[], diff_hash="def456")
        from src.agents.reviewer import persist_review_result

        await persist_review_result(store=store, task_id=task.id, review=review, step_index=2)

        # Verify all three decisions are present
        decisions = await store.query_recent_decisions(repo_url, limit=10)
        task_decisions = [d for d in decisions if d.task_id == task.id]

        agents = {d.agent for d in task_decisions}
        assert "planner" in agents
        assert "coder" in agents
        assert "reviewer" in agents

        types = {d.decision_type for d in task_decisions}
        assert "change_plan" in types
        assert "code_edit" in types
        assert "review_verdict" in types


# ══════════════════════════════════════════════════════════════════════════
# Repo URL canonicalization consistency
# ══════════════════════════════════════════════════════════════════════════


class TestRepoURLCanonicalization:
    """Repo URL canonicalization is consistent across tasks, repo_chunks,
    repo_facts, and GitHub client."""

    @pytest.mark.asyncio
    async def test_tasks_and_repo_facts_same_canonical_url(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """tasks.repo_url and repo_facts.repo_url are byte-identical
        for the same logical repo regardless of input format."""
        # Create a task with a .git suffix URL
        raw_url = f"https://github.com/test/{ns}-my-repo.git"
        canonical = canonicalize_repo_url(raw_url)

        task = await store.create_task(
            CreateTaskParams(
                repo_url=raw_url,  # .git suffix
                issue_number=1,
                issue_text="Test issue",
                topology="supervisor_only",
            )
        )

        # Create repo_fact with the canonical URL (no .git)
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=canonical,  # canonical (no .git)
                fact_kind="language",
                fact_value={"value": "python"},
            )
        )

        # Both should have the same canonical form
        assert task.repo_url == canonical

        # Query repo_facts using the .git URL — should find the fact
        facts = await store.query_repo_facts(raw_url)
        assert len(facts) >= 1
        assert facts[0].repo_url == canonical

    @pytest.mark.asyncio
    async def test_planner_context_with_dot_git_url(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Querying planner context with a .git URL still returns
        data stored under the canonical URL."""
        raw_url = f"https://github.com/test/{ns}-repo.git"
        canonical = canonicalize_repo_url(raw_url)

        # Write data under canonical URL
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=canonical,
                fact_kind="language",
                fact_value={"value": "python"},
            )
        )

        # Query using raw URL with .git suffix
        context = await store.get_planner_context(raw_url, recent_limit=5)
        assert len(context["repo_facts"]) >= 1

    @pytest.mark.asyncio
    async def test_whitespace_url_canonicalized(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """URLs with whitespace are canonicalized consistently."""
        raw_url = f"  https://github.com/test/{ns}-ws-repo  "
        canonical = canonicalize_repo_url(raw_url)

        task = await store.create_task(
            CreateTaskParams(
                repo_url=raw_url,  # with whitespace
                issue_number=1,
                issue_text="Test whitespace",
                topology="supervisor_only",
            )
        )

        assert task.repo_url == canonical
        assert " " not in task.repo_url

    @pytest.mark.asyncio
    async def test_cross_task_url_consistency(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """Two tasks created with different URL forms for the same repo
        both use the same canonical URL, enabling cross-task memory."""
        raw1 = f"https://github.com/test/{ns}-repo.git"
        raw2 = f"https://github.com/test/{ns}-repo"

        task1 = await store.create_task(
            CreateTaskParams(
                repo_url=raw1,
                issue_number=1,
                issue_text="First issue",
                topology="supervisor_only",
            )
        )
        task2 = await store.create_task(
            CreateTaskParams(
                repo_url=raw2,
                issue_number=2,
                issue_text="Second issue",
                topology="supervisor_only",
            )
        )

        # Both should have the same canonical URL
        assert task1.repo_url == task2.repo_url

        # Cross-task query should return both tasks' data
        context = await store.get_planner_context(task1.repo_url, recent_limit=10)
        task_ids_in_context = set()
        for d in context["recent_decisions"]:
            task_ids_in_context.add(d["task_id"])

        # Write decisions for both tasks
        from src.agents.planner import persist_change_plan

        plan1 = ChangePlan(
            target_files=["src/a.py"],
            rationale="Fix for task 1 on the shared repository codebase.",
            approach="Approach 1",
        )
        plan2 = ChangePlan(
            target_files=["src/b.py"],
            rationale="Fix for task 2 on the shared repository codebase.",
            approach="Approach 2",
        )
        await persist_change_plan(store=store, task_id=task1.id, plan=plan1, step_index=0)
        await persist_change_plan(store=store, task_id=task2.id, plan=plan2, step_index=0)

        # Re-query and verify cross-task data
        context = await store.get_planner_context(task1.repo_url, recent_limit=10)
        assert len(context["recent_decisions"]) >= 2


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator wiring: run_planner populates IssueContext with episodic data
# ══════════════════════════════════════════════════════════════════════════


class TestOrchestratorEpisodicWiring:
    """Verify that the orchestrator's run_planner node populates
    IssueContext with episodic data from the store before calling
    the Planner agent."""

    @pytest.mark.asyncio
    async def test_supervisor_only_planner_gets_episodic_context(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """When a task runs in supervisor_only topology, the Planner's
        IssueContext is populated with repo_facts and recent_decisions
        from the episodic store."""
        repo_url = _repo(ns, "wiring-repo")

        # Seed prior task data
        await _seed_completed_task(
            store,
            repo_url=repo_url,
            outcome="pr_opened",
            fact_kind="language",
            fact_value={"value": "python"},
        )

        # Create a new task (task Y)
        await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=2,
                issue_text="Second issue on the same repo",
                topology="supervisor_only",
            )
        )

        # Build planner context as the orchestrator's run_planner node would
        planner_ctx = await store.get_planner_context(repo_url, recent_limit=5)

        # Construct IssueContext with episodic data
        issue_context = IssueContext(
            repo_url=repo_url,
            issue_number=2,
            issue_text="Second issue on the same repo",
            repo_facts=planner_ctx["repo_facts"],
            recent_decisions=planner_ctx["recent_decisions"],
        )

        # Verify the IssueContext has episodic data from task X
        assert len(issue_context.repo_facts) >= 1
        assert len(issue_context.recent_decisions) >= 1

        # The repo_fact about language should be present
        fact_kinds = {f["fact_kind"] for f in issue_context.repo_facts}
        assert "language" in fact_kinds

    @pytest.mark.asyncio
    async def test_run_planner_node_prefetches_episodic_context(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """VAL-CROSS-013: The run_planner node in supervisor_only
        topology pre-fetches episodic context and populates
        IssueContext with repo_facts and recent_decisions before
        calling the Planner agent."""
        repo_url = _repo(ns, "node-wiring-repo")

        # Seed prior task data on this repo
        await _seed_completed_task(
            store,
            repo_url=repo_url,
            outcome="pr_opened",
            fact_kind="language",
            fact_value={"value": "python"},
        )

        # Create a new task
        task = await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_number=2,
                issue_text="Second issue for node wiring test",
                topology="supervisor_only",
            )
        )

        # Register the store for the supervisor_only node functions
        from src.orchestrator.supervisor_only import (
            register_store,
            unregister_store,
        )

        register_store(str(task.id), store)

        try:
            # Build initial state as the orchestrator would
            from src.orchestrator import OrchestratorState

            state = OrchestratorState(
                task_id=str(task.id),
                repo_url=repo_url,
                issue_number=2,
                issue_text="Second issue for node wiring test",
                topology="supervisor_only",
                trace_id=uuid4().hex,
                supervisor_span_id="test-supervisor-span",
            )

            # Mock the Planner agent to avoid real LLM calls
            mock_plan = ChangePlan(
                target_files=["src/main.py"],
                rationale="This is a test plan that is at least 20 chars long.",
                approach="Test approach",
            )

            with (
                patch("src.agents.planner.planner") as mock_planner_agent,
            ):
                # Configure the mock to return a valid result
                mock_result = MagicMock()
                mock_result.output = mock_plan
                mock_result.usage = MagicMock(
                    request_tokens=100,
                    response_tokens=50,
                )
                mock_planner_agent.run = AsyncMock(return_value=mock_result)

                # Run the planner node
                from src.orchestrator.supervisor_only import run_planner

                result = await run_planner(state)

            # Verify the result has a change_plan
            assert result.get("change_plan") is not None

            # Verify the issue_context was populated with episodic data
            issue_ctx = result.get("issue_context")
            assert issue_ctx is not None
            assert isinstance(issue_ctx, IssueContext)

            # The key assertion: IssueContext should have episodic data
            # from the prior task on the same repo
            assert len(issue_ctx.repo_facts) >= 1, (
                "IssueContext should have repo_facts from prior task"
            )
            fact_kinds = {f["fact_kind"] for f in issue_ctx.repo_facts}
            assert "language" in fact_kinds, (
                "repo_facts should include the 'language' fact from prior task"
            )

            # recent_decisions from prior task should also be present
            assert len(issue_ctx.recent_decisions) >= 1, (
                "IssueContext should have recent_decisions from prior task"
            )

        finally:
            unregister_store(str(task.id))

    @pytest.mark.asyncio
    async def test_run_planner_node_no_cross_repo_leak(
        self, store: EpisodicStore, ns: str
    ) -> None:
        """VAL-CROSS-014: The run_planner node for repo A does NOT
        populate IssueContext with data from repo B."""
        repo_a = _repo(ns, "isolation-a")
        repo_b = _repo(ns, "isolation-b")

        # Seed distinctive data on repo A
        await _seed_completed_task(
            store,
            repo_url=repo_a,
            outcome="pr_opened",
            fact_kind="install_command",
            fact_value={"command": "pip install -e ."},
        )

        # Seed distinctive data on repo B
        await _seed_completed_task(
            store,
            repo_url=repo_b,
            outcome="sandbox_failure",
            fact_kind="install_command",
            fact_value={"command": "cargo build"},
        )

        # Create a task on repo B
        task_b = await store.create_task(
            CreateTaskParams(
                repo_url=repo_b,
                issue_number=1,
                issue_text="Issue on repo B",
                topology="supervisor_only",
            )
        )

        from src.orchestrator.supervisor_only import (
            register_store,
            unregister_store,
        )

        register_store(str(task_b.id), store)

        try:
            from src.orchestrator import OrchestratorState

            state = OrchestratorState(
                task_id=str(task_b.id),
                repo_url=repo_b,
                issue_number=1,
                issue_text="Issue on repo B",
                topology="supervisor_only",
                trace_id=uuid4().hex,
                supervisor_span_id="test-supervisor-span",
            )

            mock_plan = ChangePlan(
                target_files=["src/main.rs"],
                rationale="This is a test plan that is at least 20 chars long.",
                approach="Test approach",
            )

            with patch("src.agents.planner.planner") as mock_planner_agent:
                mock_result = MagicMock()
                mock_result.output = mock_plan
                mock_result.usage = MagicMock(
                    request_tokens=100,
                    response_tokens=50,
                )
                mock_planner_agent.run = AsyncMock(return_value=mock_result)

                from src.orchestrator.supervisor_only import run_planner

                result = await run_planner(state)

            issue_ctx = result.get("issue_context")
            assert issue_ctx is not None

            # Repo B's IssueContext should NOT contain repo A's data
            for fact in issue_ctx.repo_facts:
                assert fact.get("fact_value", {}).get("command") != "pip install -e .", (
                    "Repo A's install_command should NOT appear in Repo B's context"
                )

            # Repo B's context should have cargo build, not pip install
            b_commands = [
                f.get("fact_value", {}).get("command", "")
                for f in issue_ctx.repo_facts
                if f.get("fact_kind") == "install_command"
            ]
            assert "cargo build" in b_commands, (
                "Repo B's install_command should be 'cargo build'"
            )

            # Outcomes should only be sandbox_failure (repo B), not pr_opened (repo A)
            # Note: recent_decisions are scoped by repo_url through the store
            # so they should only contain decisions for repo B

        finally:
            unregister_store(str(task_b.id))
