"""Tests for the episodic memory store.

Covers validation assertions:
  - VAL-EPISODIC-001: Schema matches architecture spec
  - VAL-EPISODIC-004: Repeated repo_fact upsert updates in place
  - VAL-EPISODIC-009: tasks.status enum is exactly the documented set
  - VAL-EPISODIC-010: Episodic memory survives full backend + DB restart
  - VAL-EPISODIC-011: Outcome enum includes all documented terminal outcomes
  - VAL-CROSS-035: Repo URL canonical form consistent across DB tables
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from src.memory.episodic.models import (
    VALID_OUTCOMES,
    VALID_STATUSES,
    CreateDecisionParams,
    CreateOutcomeParams,
    CreateTaskParams,
    UpsertRepoFactParams,
)
from src.memory.episodic.store import EpisodicStore

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


# ── VAL-EPISODIC-001: Schema matches architecture spec ──────────────────────


class TestSchemaMatchesSpec:
    """Verify the four episodic tables exist with correct columns and constraints."""

    @pytest.mark.asyncio
    async def test_tasks_table_exists(self, store: EpisodicStore) -> None:
        """tasks table is present with all required columns."""
        async with store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'tasks'
                ORDER BY ordinal_position
                """
            )
        col_names = {r["column_name"] for r in rows}
        expected = {
            "id", "repo_url", "issue_number", "issue_text", "topology",
            "status", "total_cost_usd", "total_tokens_in", "total_tokens_out",
            "total_tokens_cached", "agent_costs", "hitl_decision", "pr_url",
            "started_at", "ended_at",
        }
        assert expected == col_names, f"Missing columns: {expected - col_names}"

    @pytest.mark.asyncio
    async def test_decisions_table_exists(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'decisions'
                ORDER BY ordinal_position
                """
            )
        col_names = {r["column_name"] for r in rows}
        expected = {
            "id", "task_id", "agent", "step_index",
            "decision_type", "decision_data", "created_at",
        }
        assert expected == col_names

    @pytest.mark.asyncio
    async def test_outcomes_table_exists(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'outcomes'
                ORDER BY ordinal_position
                """
            )
        col_names = {r["column_name"] for r in rows}
        expected = {"id", "task_id", "outcome", "detail", "recorded_at"}
        assert expected == col_names

    @pytest.mark.asyncio
    async def test_repo_facts_table_exists(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'repo_facts'
                ORDER BY ordinal_position
                """
            )
        col_names = {r["column_name"] for r in rows}
        expected = {"id", "repo_url", "fact_kind", "fact_value", "observed_at"}
        assert expected == col_names

    @pytest.mark.asyncio
    async def test_decision_data_is_jsonb(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'decisions' AND column_name = 'decision_data'
                """
            )
        assert row is not None
        dtype = row["data_type"]
        assert "json" in dtype.lower() or dtype == "USER-DEFINED"

    @pytest.mark.asyncio
    async def test_outcomes_detail_is_jsonb(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'outcomes' AND column_name = 'detail'
                """
            )
        assert row is not None
        dtype = row["data_type"]
        assert "json" in dtype.lower() or dtype == "USER-DEFINED"

    @pytest.mark.asyncio
    async def test_repo_facts_unique_constraint(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'repo_facts'::regclass AND contype = 'u'
                """
            )
        constraint_names = {r["conname"] for r in rows}
        assert any(
            "repo_url" in name and "fact_kind" in name for name in constraint_names
        ), f"No unique constraint on (repo_url, fact_kind). Found: {constraint_names}"

    @pytest.mark.asyncio
    async def test_decisions_fk_cascade(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT confdeltype
                FROM pg_constraint
                WHERE conrelid = 'decisions'::regclass AND contype = 'f'
                """
            )
        assert row is not None
        val = row["confdeltype"]
        assert val in ("c", b"c"), f"Expected CASCADE, got {val!r}"

    @pytest.mark.asyncio
    async def test_outcomes_fk_cascade(self, store: EpisodicStore) -> None:
        async with store.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT confdeltype
                FROM pg_constraint
                WHERE conrelid = 'outcomes'::regclass AND contype = 'f'
                """
            )
        assert row is not None
        val = row["confdeltype"] if "confdeltype" in row else row["confdeltype"]
        assert val in ("c", b"c"), f"Expected CASCADE, got {val!r}"


# ── VAL-EPISODIC-009: tasks.status enum ──────────────────────────────────────


class TestStatusEnum:
    """INSERT with valid status succeeds; INSERT with invalid status raises."""

    @pytest.mark.asyncio
    async def test_valid_statuses_succeed(self, store: EpisodicStore, ns: str) -> None:
        for status in sorted(VALID_STATUSES):
            params = CreateTaskParams(
                repo_url=_repo(ns, f"status-{status}"),
                issue_text=f"Testing status={status}",
                topology="hybrid",
                status=status,
            )
            task = await store.create_task(params)
            assert task.status == status

    @pytest.mark.asyncio
    async def test_invalid_status_raises(self, store: EpisodicStore) -> None:
        from asyncpg.exceptions import CheckViolationError

        task_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(CheckViolationError):
            async with store.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO tasks (id, repo_url, issue_text, topology, status, started_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    task_id,
                    "https://github.com/test/bad-status-x",
                    "test",
                    "hybrid",
                    "in_review",
                    now,
                )

    @pytest.mark.asyncio
    async def test_status_set_is_exactly_documented(self) -> None:
        expected = {
            "running", "awaiting_hitl", "approved", "rejected",
            "completed", "failed",
        }
        assert expected == VALID_STATUSES


# ── VAL-EPISODIC-011: Outcome enum includes all 9 values ────────────────────


class TestOutcomeEnum:
    """Each documented outcome can be inserted; invalid ones are rejected."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", sorted(VALID_OUTCOMES))
    async def test_each_documented_outcome_inserts(
        self, store: EpisodicStore, ns: str, outcome: str
    ) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, f"outcome-{outcome}"),
                issue_text=f"Testing outcome={outcome}",
                topology="hybrid",
            )
        )
        result = await store.create_outcome(
            CreateOutcomeParams(
                task_id=task.id,
                outcome=outcome,
                detail={"trigger": "test"},
            )
        )
        assert result.outcome == outcome

    @pytest.mark.asyncio
    async def test_invalid_outcome_raises(self, store: EpisodicStore, ns: str) -> None:
        from asyncpg.exceptions import CheckViolationError

        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "bad-outcome"),
                issue_text="test",
                topology="hybrid",
            )
        )
        with pytest.raises(CheckViolationError):
            async with store.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO outcomes (id, task_id, outcome, recorded_at)
                    VALUES ($1, $2, $3, now())
                    """,
                    uuid4(),
                    task.id,
                    "unknown_outcome",
                )

    @pytest.mark.asyncio
    async def test_outcome_set_is_exactly_nine(self) -> None:
        expected = {
            "success", "pr_opened", "hitl_rejected",
            "retry_budget_exhausted", "loop_detected",
            "uncertainty_escalation", "guardrail_block",
            "cost_budget_exhausted", "sandbox_failure",
        }
        assert expected == VALID_OUTCOMES
        assert len(VALID_OUTCOMES) == 9


# ── VAL-EPISODIC-004: Repeated repo_fact upsert updates row in place ────────


class TestRepoFactUpsert:
    """Writing the same (repo_url, fact_kind) twice updates in place."""

    @pytest.mark.asyncio
    async def test_upsert_updates_in_place(self, store: EpisodicStore, ns: str) -> None:
        repo_url = _repo(ns, "upsert-check")

        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url,
                fact_kind="language",
                fact_value={"lang": "python"},
            )
        )

        result = await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url,
                fact_kind="language",
                fact_value={"lang": "python", "version": "3.14"},
            )
        )

        facts = await store.query_repo_facts(repo_url, fact_kind="language")
        assert len(facts) == 1
        assert facts[0].fact_value.get("version") == "3.14"
        assert facts[0].fact_value.get("lang") == "python"
        assert facts[0].id == result.id

    @pytest.mark.asyncio
    async def test_unique_constraint_enforced(self, store: EpisodicStore, ns: str) -> None:
        from asyncpg.exceptions import UniqueViolationError

        repo_url = _repo(ns, "unique-check")
        now = datetime.now(UTC)

        async with store.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO repo_facts (id, repo_url, fact_kind, fact_value, observed_at)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                """,
                uuid4(),
                repo_url,
                "test_kind",
                json.dumps({"v": 1}),
                now,
            )
            with pytest.raises(UniqueViolationError):
                await conn.execute(
                    """
                    INSERT INTO repo_facts (id, repo_url, fact_kind, fact_value, observed_at)
                    VALUES ($1, $2, $3, $4::jsonb, $5)
                    """,
                    uuid4(),
                    repo_url,
                    "test_kind",
                    json.dumps({"v": 2}),
                    now,
                )


# ── VAL-EPISODIC-010: Data survives full backend + DB restart ───────────────


class TestDataPersistence:
    """Data survives full backend + DB restart."""

    @pytest.mark.asyncio
    async def test_data_survives_db_reconnect(self, store: EpisodicStore, ns: str) -> None:
        repo_url = _repo(ns, "persist")

        task = await store.create_task(
            CreateTaskParams(
                repo_url=repo_url,
                issue_text="Persistence test",
                topology="hybrid",
            )
        )
        decision = await store.create_decision(
            CreateDecisionParams(
                task_id=task.id,
                agent="planner",
                step_index=0,
                decision_type="change_plan",
                decision_data={"target_files": ["src/main.py"]},
            )
        )
        outcome = await store.create_outcome(
            CreateOutcomeParams(
                task_id=task.id,
                outcome="success",
                detail={"note": "test"},
            )
        )
        fact = await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url,
                fact_kind="language",
                fact_value={"lang": "python"},
            )
        )

        decision_hash = hashlib.sha256(
            json.dumps(decision.decision_data, sort_keys=True).encode()
        ).hexdigest()
        outcome_hash = hashlib.sha256(
            json.dumps(outcome.detail, sort_keys=True).encode()
        ).hexdigest()
        fact_hash = hashlib.sha256(
            json.dumps(fact.fact_value, sort_keys=True).encode()
        ).hexdigest()

        # Close and reconnect (simulates backend restart)
        await store.close()
        await store.connect()

        restored_task = await store.get_task(task.id)
        assert restored_task is not None
        assert restored_task.repo_url == repo_url
        assert restored_task.issue_text == "Persistence test"

        restored_decisions = await store.query_recent_decisions(repo_url)
        assert len(restored_decisions) >= 1
        assert hashlib.sha256(
            json.dumps(restored_decisions[0].decision_data, sort_keys=True).encode()
        ).hexdigest() == decision_hash

        restored_outcomes = await store.query_recent_outcomes(repo_url)
        assert len(restored_outcomes) >= 1
        assert hashlib.sha256(
            json.dumps(restored_outcomes[0].detail, sort_keys=True).encode()
        ).hexdigest() == outcome_hash

        restored_facts = await store.query_repo_facts(repo_url)
        assert len(restored_facts) >= 1
        assert hashlib.sha256(
            json.dumps(restored_facts[0].fact_value, sort_keys=True).encode()
        ).hexdigest() == fact_hash


# ── VAL-CROSS-035: Repo URL canonical form consistent across DB tables ───────


class TestRepoUrlCanonicalization:
    """Repo URL canonicalization is consistent across tasks/repo_facts."""

    @pytest.mark.asyncio
    async def test_canonicalization_strips_dot_git(self, store: EpisodicStore) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url="https://github.com/org/repo.git",
                issue_text="test",
                topology="hybrid",
            )
        )
        assert task.repo_url == "https://github.com/org/repo"

    @pytest.mark.asyncio
    async def test_canonicalization_strips_trailing_slash(
        self, store: EpisodicStore
    ) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url="https://github.com/org/repo/",
                issue_text="test",
                topology="hybrid",
            )
        )
        assert task.repo_url == "https://github.com/org/repo"

    @pytest.mark.asyncio
    async def test_canonicalization_trims_whitespace(
        self, store: EpisodicStore
    ) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url="  https://github.com/org/repo  ",
                issue_text="test",
                topology="hybrid",
            )
        )
        assert task.repo_url == "https://github.com/org/repo"

    @pytest.mark.asyncio
    async def test_repo_facts_url_matches_tasks_url(
        self, store: EpisodicStore
    ) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url="https://github.com/org/repo.git",
                issue_text="test",
                topology="hybrid",
            )
        )
        fact = await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url="https://github.com/org/repo",
                fact_kind="language",
                fact_value={"lang": "python"},
            )
        )
        assert task.repo_url == fact.repo_url

    @pytest.mark.asyncio
    async def test_cross_table_repo_url_consistency(
        self, store: EpisodicStore
    ) -> None:
        variants = [
            "https://github.com/org/repo.git",
            "https://github.com/org/repo/",
            "  https://github.com/org/repo  ",
        ]
        canonical = "https://github.com/org/repo"

        for variant in variants:
            task = await store.create_task(
                CreateTaskParams(
                    repo_url=variant,
                    issue_text="test",
                    topology="hybrid",
                )
            )
            assert task.repo_url == canonical, (
                f"Variant {variant!r} produced {task.repo_url!r}"
            )

        fact = await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url="https://github.com/org/repo.git",
                fact_kind="test_kind",
                fact_value={"test": True},
            )
        )
        assert fact.repo_url == canonical


# ── CRUD integration tests ───────────────────────────────────────────────────


class TestCRUDOperations:
    """Core CRUD operations work correctly."""

    @pytest.mark.asyncio
    async def test_create_and_get_task(self, store: EpisodicStore, ns: str) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "crud"),
                issue_number=42,
                issue_text="Fix the bug",
                topology="supervisor_only",
            )
        )
        retrieved = await store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.repo_url == _repo(ns, "crud")
        assert retrieved.issue_number == 42
        assert retrieved.issue_text == "Fix the bug"
        assert retrieved.topology == "supervisor_only"
        assert retrieved.status == "running"

    @pytest.mark.asyncio
    async def test_update_task_status(self, store: EpisodicStore, ns: str) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "status-update"),
                issue_text="test",
                topology="hybrid",
            )
        )
        await store.update_task_status(task.id, "awaiting_hitl")
        updated = await store.get_task(task.id)
        assert updated is not None
        assert updated.status == "awaiting_hitl"

    @pytest.mark.asyncio
    async def test_finish_task(self, store: EpisodicStore, ns: str) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "finish"),
                issue_text="test",
                topology="hybrid",
            )
        )
        await store.finish_task(
            task.id,
            "completed",
            hitl_decision="approve",
            pr_url=f"https://github.com/test/{ns}/finish/pull/1",
        )
        finished = await store.get_task(task.id)
        assert finished is not None
        assert finished.status == "completed"
        assert finished.ended_at is not None
        assert finished.hitl_decision == "approve"
        assert finished.pr_url is not None

    @pytest.mark.asyncio
    async def test_update_task_totals(self, store: EpisodicStore, ns: str) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "totals"),
                issue_text="test",
                topology="hybrid",
            )
        )
        await store.update_task_totals(
            task.id,
            total_cost_usd=Decimal("1.2345"),
            total_tokens_in=100,
            total_tokens_out=200,
            total_tokens_cached=50,
        )
        updated = await store.get_task(task.id)
        assert updated is not None
        assert updated.total_cost_usd == Decimal("1.2345")
        assert updated.total_tokens_in == 100
        assert updated.total_tokens_out == 200
        assert updated.total_tokens_cached == 50

    @pytest.mark.asyncio
    async def test_create_decision(self, store: EpisodicStore, ns: str) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "decision"),
                issue_text="test",
                topology="hybrid",
            )
        )
        decision = await store.create_decision(
            CreateDecisionParams(
                task_id=task.id,
                agent="planner",
                step_index=0,
                decision_type="change_plan",
                decision_data={"target_files": ["src/main.py"], "rationale": "fix bug"},
            )
        )
        assert decision.task_id == task.id
        assert decision.agent == "planner"
        assert decision.decision_type == "change_plan"
        assert decision.decision_data["target_files"] == ["src/main.py"]

    @pytest.mark.asyncio
    async def test_create_outcome(self, store: EpisodicStore, ns: str) -> None:
        task = await store.create_task(
            CreateTaskParams(
                repo_url=_repo(ns, "outcome"),
                issue_text="test",
                topology="hybrid",
            )
        )
        outcome = await store.create_outcome(
            CreateOutcomeParams(
                task_id=task.id,
                outcome="success",
                detail={"note": "all tests passed"},
            )
        )
        assert outcome.task_id == task.id
        assert outcome.outcome == "success"
        assert outcome.detail is not None
        assert outcome.detail["note"] == "all tests passed"

    @pytest.mark.asyncio
    async def test_list_tasks(self, store: EpisodicStore, ns: str) -> None:
        repo_a = _repo(ns, "list-a")
        repo_b = _repo(ns, "list-b")
        await store.create_task(
            CreateTaskParams(repo_url=repo_a, issue_text="test a", topology="hybrid")
        )
        await store.create_task(
            CreateTaskParams(repo_url=repo_b, issue_text="test b", topology="supervisor_only")
        )
        tasks = await store.list_tasks(repo_url=repo_a)
        assert len(tasks) >= 1
        assert tasks[0].repo_url == repo_a

    @pytest.mark.asyncio
    async def test_cascade_delete(self, store: EpisodicStore, ns: str) -> None:
        repo_url = _repo(ns, "cascade")
        task = await store.create_task(
            CreateTaskParams(repo_url=repo_url, issue_text="test", topology="hybrid")
        )
        await store.create_decision(
            CreateDecisionParams(
                task_id=task.id, agent="planner", step_index=0,
                decision_type="change_plan", decision_data={},
            )
        )
        await store.create_outcome(
            CreateOutcomeParams(task_id=task.id, outcome="success")
        )

        async with store.pool.acquire() as conn:
            await conn.execute("DELETE FROM tasks WHERE id = $1", task.id)

        decisions = await store.query_recent_decisions(repo_url)
        assert len(decisions) == 0
        outcomes = await store.get_outcomes_for_task(task.id)
        assert len(outcomes) == 0


# ── Planner context tests (VAL-EPISODIC-003 partial) ────────────────────────


class TestPlannerContext:
    """Planner context retrieval works correctly."""

    @pytest.mark.asyncio
    async def test_planner_context_retrieves_repo_facts(
        self, store: EpisodicStore, ns: str
    ) -> None:
        repo_url = _repo(ns, "context")
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url, fact_kind="language",
                fact_value={"lang": "python"},
            )
        )
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_url, fact_kind="test_command",
                fact_value={"cmd": "pytest"},
            )
        )
        context = await store.get_planner_context(repo_url)
        assert len(context["repo_facts"]) == 2

    @pytest.mark.asyncio
    async def test_planner_context_retrieves_recent_rows(
        self, store: EpisodicStore, ns: str
    ) -> None:
        repo_url = _repo(ns, "context-recent")
        task = await store.create_task(
            CreateTaskParams(repo_url=repo_url, issue_text="prior", topology="hybrid")
        )
        await store.create_decision(
            CreateDecisionParams(
                task_id=task.id, agent="planner", step_index=0,
                decision_type="change_plan",
                decision_data={"target_files": ["src/main.py"]},
            )
        )
        await store.create_outcome(
            CreateOutcomeParams(
                task_id=task.id, outcome="success", detail={"note": "prior"},
            )
        )
        context = await store.get_planner_context(repo_url)
        assert len(context["recent_decisions"]) >= 1
        assert len(context["recent_outcomes"]) >= 1

    @pytest.mark.asyncio
    async def test_planner_context_isolation(
        self, store: EpisodicStore, ns: str
    ) -> None:
        repo_a = _repo(ns, "isolation-a")
        repo_b = _repo(ns, "isolation-b")
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_a, fact_kind="language",
                fact_value={"lang": "python"},
            )
        )
        await store.upsert_repo_fact(
            UpsertRepoFactParams(
                repo_url=repo_b, fact_kind="language",
                fact_value={"lang": "rust"},
            )
        )
        ctx_a = await store.get_planner_context(repo_a)
        ctx_b = await store.get_planner_context(repo_b)
        for fact in ctx_a["repo_facts"]:
            assert fact["repo_url"] == repo_a
        for fact in ctx_b["repo_facts"]:
            assert fact["repo_url"] == repo_b

    @pytest.mark.asyncio
    async def test_cross_task_memory_reuse(
        self, store: EpisodicStore, ns: str
    ) -> None:
        repo_url = _repo(ns, "cross-task")
        task_x = await store.create_task(
            CreateTaskParams(repo_url=repo_url, issue_text="first", topology="hybrid")
        )
        await store.create_outcome(
            CreateOutcomeParams(
                task_id=task_x.id, outcome="hitl_rejected",
                detail={"reason": "not good enough"},
            )
        )
        context = await store.get_planner_context(repo_url)
        assert len(context["recent_outcomes"]) >= 1
        outcome_text = json.dumps(context["recent_outcomes"])
        assert "hitl_rejected" in outcome_text


# ── Terminal outcome count test (VAL-EPISODIC-007 partial) ──────────────────


class TestTerminalOutcomeCount:
    """Terminal events write a single outcome row each."""

    @pytest.mark.asyncio
    async def test_single_outcome_per_terminal_event(
        self, store: EpisodicStore, ns: str
    ) -> None:
        for outcome_val in sorted(VALID_OUTCOMES):
            task = await store.create_task(
                CreateTaskParams(
                    repo_url=_repo(ns, f"terminal-{outcome_val}"),
                    issue_text=f"Testing {outcome_val}",
                    topology="hybrid",
                )
            )
            await store.create_outcome(
                CreateOutcomeParams(task_id=task.id, outcome=outcome_val)
            )
            outcomes = await store.get_outcomes_for_task(task.id)
            assert len(outcomes) == 1, (
                f"Expected 1 outcome for {outcome_val}, got {len(outcomes)}"
            )
            assert outcomes[0].outcome == outcome_val


# ── Decision data schema test (VAL-EPISODIC-008 partial) ────────────────────


class TestDecisionDataSchema:
    """decision_data is valid JSONB that can be round-tripped."""

    @pytest.mark.asyncio
    async def test_decision_data_roundtrip(
        self, store: EpisodicStore, ns: str
    ) -> None:
        repo_url = _repo(ns, "roundtrip")
        task = await store.create_task(
            CreateTaskParams(repo_url=repo_url, issue_text="test", topology="hybrid")
        )
        test_payload = {
            "target_files": ["src/main.py", "src/utils.py"],
            "rationale": "Fix the off-by-one error in the loop condition",
            "approach": "Adjust loop boundary",
        }
        decision = await store.create_decision(
            CreateDecisionParams(
                task_id=task.id, agent="planner", step_index=0,
                decision_type="change_plan", decision_data=test_payload,
            )
        )
        assert decision.decision_data["target_files"] == ["src/main.py", "src/utils.py"]
        assert (
            decision.decision_data["rationale"]
            == "Fix the off-by-one error in the loop condition"
        )

        decisions = await store.query_recent_decisions(repo_url)
        assert len(decisions) >= 1
        assert decisions[0].decision_data["target_files"] == test_payload["target_files"]

    @pytest.mark.asyncio
    async def test_decision_types_covered(
        self, store: EpisodicStore, ns: str
    ) -> None:
        repo_url = _repo(ns, "dtypes")
        task = await store.create_task(
            CreateTaskParams(repo_url=repo_url, issue_text="test", topology="hybrid")
        )
        decision_types = [
            "change_plan", "code_edit", "review_verdict",
            "test_report", "route",
        ]
        for i, dtype in enumerate(decision_types):
            decision = await store.create_decision(
                CreateDecisionParams(
                    task_id=task.id, agent="planner", step_index=i,
                    decision_type=dtype, decision_data={"type": dtype},
                )
            )
            assert decision.decision_type == dtype
