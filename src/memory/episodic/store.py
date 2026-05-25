"""Episodic memory store — async Postgres-backed CRUD operations.

All write operations are scoped to the orchestrator layer.
Agents should only consume read methods (e.g. ``query_repo_facts``,
``query_recent_decisions``, ``query_recent_outcomes``).

Repo URL canonicalization is applied uniformly on every write so
that ``tasks.repo_url``, ``repo_chunks.repo_url``, and
``repo_facts.repo_url`` are byte-identical for the same logical repo
(VAL-CROSS-035).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from src.github_client.client import canonicalize_repo_url
from src.memory.episodic.models import (
    VALID_STATUSES,
    CreateDecisionParams,
    CreateOutcomeParams,
    CreateTaskParams,
    DecisionRow,
    OutcomeRow,
    RepoFactRow,
    TaskRow,
    UpsertRepoFactParams,
)
from src.memory.episodic.schema import EPISODIC_SCHEMA_SQL

logger = logging.getLogger(__name__)


def _dsn() -> str:
    """Build Postgres DSN from environment variables."""
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
    )


class EpisodicStore:
    """Async episodic memory store backed by Postgres.

    Usage::

        store = EpisodicStore()
        await store.connect()
        # ... use store ...
        await store.close()

    Or as an async context manager::

        async with EpisodicStore() as store:
            task = await store.create_task(...)
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or _dsn()
        self._pool: asyncpg.Pool | None = None

    # ── Connection lifecycle ──────────────────────────────────────

    async def connect(self) -> None:
        """Open a connection pool and ensure schema exists."""
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        await self._ensure_schema()

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> EpisodicStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ── Schema ────────────────────────────────────────────────────

    async def _ensure_schema(self) -> None:
        """Create episodic tables if they do not exist."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(EPISODIC_SCHEMA_SQL)

    @property
    def pool(self) -> asyncpg.Pool:
        """Return the connection pool; raises if not connected."""
        if self._pool is None:
            msg = "EpisodicStore not connected. Call await store.connect() first."
            raise RuntimeError(msg)
        return self._pool

    # ── Tasks ─────────────────────────────────────────────────────

    async def create_task(self, params: CreateTaskParams) -> TaskRow:
        """INSERT a new task row.  Returns the persisted row.

        Repo URL is canonicalized before storage.
        Status is validated at both the Pydantic and DB level.
        """
        task_id = uuid4()
        canon_url = canonicalize_repo_url(params.repo_url)
        now = datetime.now(UTC)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tasks
                    (id, repo_url, issue_number, issue_text, topology, status, started_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                task_id,
                canon_url,
                params.issue_number,
                params.issue_text,
                params.topology,
                params.status,
                now,
            )

        assert row is not None, "INSERT RETURNING * should always produce a row"
        return self._task_row_from_record(row)

    async def get_task(self, task_id: UUID) -> TaskRow | None:
        """SELECT a task by id.  Returns None if not found."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)

        if row is None:
            return None
        return self._task_row_from_record(row)

    async def update_task_status(self, task_id: UUID, status: str) -> None:
        """Update ``tasks.status``.  Raises ValueError for invalid status."""
        if status not in VALID_STATUSES:
            msg = f"Invalid status {status!r}"
            raise ValueError(msg)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET status = $2 WHERE id = $1",
                task_id,
                status,
            )

    async def update_task_totals(
        self,
        task_id: UUID,
        *,
        total_cost_usd: Decimal | None = None,
        total_tokens_in: int | None = None,
        total_tokens_out: int | None = None,
        total_tokens_cached: int | None = None,
        agent_costs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Update running totals on a task row.

        agent_costs maps agent name → {tokens_in, tokens_out, cached_tokens, cost_usd}.
        """
        async with self.pool.acquire() as conn:
            if total_cost_usd is not None:
                await conn.execute(
                    "UPDATE tasks SET total_cost_usd = $2 WHERE id = $1",
                    task_id,
                    total_cost_usd,
                )
            if total_tokens_in is not None:
                await conn.execute(
                    "UPDATE tasks SET total_tokens_in = $2 WHERE id = $1",
                    task_id,
                    total_tokens_in,
                )
            if total_tokens_out is not None:
                await conn.execute(
                    "UPDATE tasks SET total_tokens_out = $2 WHERE id = $1",
                    task_id,
                    total_tokens_out,
                )
            if total_tokens_cached is not None:
                await conn.execute(
                    "UPDATE tasks SET total_tokens_cached = $2 WHERE id = $1",
                    task_id,
                    total_tokens_cached,
                )
            if agent_costs is not None:
                import json

                await conn.execute(
                    "UPDATE tasks SET agent_costs = $2 WHERE id = $1",
                    task_id,
                    json.dumps(agent_costs),
                )

    async def finish_task(
        self,
        task_id: UUID,
        status: str,
        *,
        hitl_decision: str | None = None,
        pr_url: str | None = None,
    ) -> None:
        """Mark a task as finished: set status, ended_at, optional fields."""
        if status not in VALID_STATUSES:
            msg = f"Invalid status {status!r}"
            raise ValueError(msg)
        now = datetime.now(UTC)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE tasks
                SET status = $2, ended_at = $3, hitl_decision = $4, pr_url = $5
                WHERE id = $1
                """,
                task_id,
                status,
                now,
                hitl_decision,
                pr_url,
            )

    async def list_tasks(
        self,
        *,
        repo_url: str | None = None,
        status: str | None = None,
        outcome: str | None = None,
        topology: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskRow]:
        """List tasks with optional filters."""
        conditions: list[str] = []
        args: list[Any] = []
        idx = 1

        if repo_url is not None:
            canon_url = canonicalize_repo_url(repo_url)
            conditions.append(f"repo_url = ${idx}")
            args.append(canon_url)
            idx += 1

        if status is not None:
            # Support comma-separated status values (e.g. "completed,failed,rejected")
            status_values = [s.strip() for s in status.split(",") if s.strip()]
            if len(status_values) == 1:
                conditions.append(f"status = ${idx}")
                args.append(status_values[0])
                idx += 1
            elif status_values:
                placeholders = ", ".join(f"${idx + i}" for i in range(len(status_values)))
                conditions.append(f"status IN ({placeholders})")
                args.extend(status_values)
                idx += len(status_values)

        if topology is not None:
            # Support comma-separated topology values
            topology_values = [t.strip() for t in topology.split(",") if t.strip()]
            if len(topology_values) == 1:
                conditions.append(f"topology = ${idx}")
                args.append(topology_values[0])
                idx += 1
            elif topology_values:
                placeholders = ", ".join(f"${idx + i}" for i in range(len(topology_values)))
                conditions.append(f"topology IN ({placeholders})")
                args.extend(topology_values)
                idx += len(topology_values)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Filter by outcome requires a join
        if outcome is not None:
            query = f"""
                SELECT DISTINCT t.*
                FROM tasks t
                JOIN outcomes o ON o.task_id = t.id
                {where}
                {"AND" if conditions else "WHERE"} o.outcome = ${idx}
                ORDER BY t.ended_at DESC NULLS LAST, t.started_at DESC
                LIMIT ${idx + 1} OFFSET ${idx + 2}
            """
            args.extend([outcome, limit, offset])
        else:
            query = f"""
                SELECT * FROM tasks
                {where}
                ORDER BY ended_at DESC NULLS LAST, started_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}
            """
            args.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._task_row_from_record(r) for r in rows]

    async def get_latest_outcomes(
        self, task_ids: list[UUID]
    ) -> dict[UUID, str]:
        """Return the latest outcome string for each task_id.

        Returns a dict mapping task_id → outcome.  Tasks without
        an outcomes row are absent from the dict.
        """
        if not task_ids:
            return {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (o.task_id)
                    o.task_id, o.outcome
                FROM outcomes o
                WHERE o.task_id = ANY($1)
                ORDER BY o.task_id, o.recorded_at DESC
                """,
                task_ids,
            )
        return {r["task_id"]: r["outcome"] for r in rows}

    # ── Decisions ─────────────────────────────────────────────────

    async def create_decision(self, params: CreateDecisionParams) -> DecisionRow:
        """INSERT a decision row."""
        decision_id = uuid4()
        now = datetime.now(UTC)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO decisions
                    (id, task_id, agent, step_index, decision_type, decision_data, created_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                RETURNING *
                """,
                decision_id,
                params.task_id,
                params.agent,
                params.step_index,
                params.decision_type,
                _dict_to_json(params.decision_data),
                now,
            )

        assert row is not None, "INSERT RETURNING * should always produce a row"
        return self._decision_row_from_record(row)

    async def get_decisions_for_task(self, task_id: UUID) -> list[DecisionRow]:
        """Return all decisions for a given task, ordered by step_index."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM decisions WHERE task_id = $1 ORDER BY step_index, created_at",
                task_id,
            )
        return [self._decision_row_from_record(r) for r in rows]

    async def query_recent_decisions(
        self,
        repo_url: str,
        *,
        limit: int = 5,
    ) -> list[DecisionRow]:
        """Return the most recent N decisions for tasks on the given repo_url.

        Repo URL is canonicalized for consistent scoping.
        """
        canon_url = canonicalize_repo_url(repo_url)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.*
                FROM decisions d
                JOIN tasks t ON t.id = d.task_id
                WHERE t.repo_url = $1
                ORDER BY d.created_at DESC
                LIMIT $2
                """,
                canon_url,
                limit,
            )

        return [self._decision_row_from_record(r) for r in rows]

    # ── Outcomes ──────────────────────────────────────────────────

    async def create_outcome(self, params: CreateOutcomeParams) -> OutcomeRow:
        """INSERT an outcome row."""
        outcome_id = uuid4()
        now = datetime.now(UTC)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO outcomes
                    (id, task_id, outcome, detail, recorded_at)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING *
                """,
                outcome_id,
                params.task_id,
                params.outcome,
                _dict_to_json(params.detail),
                now,
            )

        assert row is not None, "INSERT RETURNING * should always produce a row"
        return self._outcome_row_from_record(row)

    async def query_recent_outcomes(
        self,
        repo_url: str,
        *,
        limit: int = 5,
    ) -> list[OutcomeRow]:
        """Return the most recent N outcomes for tasks on the given repo_url.

        Repo URL is canonicalized for consistent scoping.
        """
        canon_url = canonicalize_repo_url(repo_url)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT o.*
                FROM outcomes o
                JOIN tasks t ON t.id = o.task_id
                WHERE t.repo_url = $1
                ORDER BY o.recorded_at DESC
                LIMIT $2
                """,
                canon_url,
                limit,
            )

        return [self._outcome_row_from_record(r) for r in rows]

    async def get_outcomes_for_task(self, task_id: UUID) -> list[OutcomeRow]:
        """Return all outcomes for a given task."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM outcomes WHERE task_id = $1 ORDER BY recorded_at",
                task_id,
            )

        return [self._outcome_row_from_record(r) for r in rows]

    # ── Repo Facts ───────────────────────────────────────────────

    async def upsert_repo_fact(self, params: UpsertRepoFactParams) -> RepoFactRow:
        """Upsert a repo_fact row.  If (repo_url, fact_kind) already exists,
        update ``fact_value`` and ``observed_at`` in place (VAL-EPISODIC-004).

        Repo URL is canonicalized before storage.
        """
        fact_id = uuid4()
        canon_url = canonicalize_repo_url(params.repo_url)
        now = datetime.now(UTC)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO repo_facts
                    (id, repo_url, fact_kind, fact_value, observed_at)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                ON CONFLICT (repo_url, fact_kind)
                DO UPDATE SET
                    fact_value = EXCLUDED.fact_value,
                    observed_at = EXCLUDED.observed_at
                RETURNING *
                """,
                fact_id,
                canon_url,
                params.fact_kind,
                _dict_to_json(params.fact_value),
                now,
            )

        assert row is not None, "INSERT RETURNING * should always produce a row"
        return self._repo_fact_row_from_record(row)

    async def query_repo_facts(
        self,
        repo_url: str,
        *,
        fact_kind: str | None = None,
    ) -> list[RepoFactRow]:
        """Query repo_facts for a given repo_url, optionally filtered by kind.

        Repo URL is canonicalized for consistent scoping.
        """
        canon_url = canonicalize_repo_url(repo_url)
        if fact_kind is not None:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM repo_facts WHERE repo_url = $1 AND fact_kind = $2",
                    canon_url,
                    fact_kind,
                )
        else:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM repo_facts WHERE repo_url = $1",
                    canon_url,
                )

        return [self._repo_fact_row_from_record(r) for r in rows]

    # ── Planner context helper ───────────────────────────────────

    async def get_planner_context(
        self,
        repo_url: str,
        *,
        recent_limit: int = 5,
    ) -> dict[str, Any]:
        """Build the Planner's episodic context for a given repo_url.

        Returns a dict with:
          - ``repo_facts``: list of RepoFactRow
          - ``recent_decisions``: list of DecisionRow
          - ``recent_outcomes``: list of OutcomeRow
        """
        facts = await self.query_repo_facts(repo_url)
        decisions = await self.query_recent_decisions(repo_url, limit=recent_limit)
        outcomes = await self.query_recent_outcomes(repo_url, limit=recent_limit)

        return {
            "repo_facts": [f.model_dump(mode="json") for f in facts],
            "recent_decisions": [d.model_dump(mode="json") for d in decisions],
            "recent_outcomes": [o.model_dump(mode="json") for o in outcomes],
        }

    # ── Row mappers ───────────────────────────────────────────────

    @staticmethod
    def _task_row_from_record(record: asyncpg.Record) -> TaskRow:
        """Map an asyncpg Record to a TaskRow."""
        return TaskRow(
            id=record["id"],
            repo_url=record["repo_url"],
            issue_number=record["issue_number"],
            issue_text=record["issue_text"],
            topology=record["topology"],
            status=record["status"],
            total_cost_usd=record["total_cost_usd"],
            total_tokens_in=record["total_tokens_in"],
            total_tokens_out=record["total_tokens_out"],
            total_tokens_cached=record["total_tokens_cached"],
            agent_costs=record.get("agent_costs"),
            hitl_decision=record["hitl_decision"],
            pr_url=record["pr_url"],
            started_at=record["started_at"],
            ended_at=record["ended_at"],
        )

    @staticmethod
    def _decision_row_from_record(record: asyncpg.Record) -> DecisionRow:
        """Map an asyncpg Record to a DecisionRow."""
        data = record["decision_data"]
        if isinstance(data, str):
            import json
            data = json.loads(data)
        return DecisionRow(
            id=record["id"],
            task_id=record["task_id"],
            agent=record["agent"],
            step_index=record["step_index"],
            decision_type=record["decision_type"],
            decision_data=data,
            created_at=record["created_at"],
        )

    @staticmethod
    def _outcome_row_from_record(record: asyncpg.Record) -> OutcomeRow:
        """Map an asyncpg Record to an OutcomeRow."""
        detail = record["detail"]
        if isinstance(detail, str):
            import json
            detail = json.loads(detail)
        return OutcomeRow(
            id=record["id"],
            task_id=record["task_id"],
            outcome=record["outcome"],
            detail=detail,
            recorded_at=record["recorded_at"],
        )

    @staticmethod
    def _repo_fact_row_from_record(record: asyncpg.Record) -> RepoFactRow:
        """Map an asyncpg Record to a RepoFactRow."""
        value = record["fact_value"]
        if isinstance(value, str):
            import json
            value = json.loads(value)
        return RepoFactRow(
            id=record["id"],
            repo_url=record["repo_url"],
            fact_kind=record["fact_kind"],
            fact_value=value,
            observed_at=record["observed_at"],
        )


def _dict_to_json(val: dict[str, Any] | None) -> str | None:
    """Serialize a dict to JSON string for JSONB insertion."""
    if val is None:
        return None
    import json
    return json.dumps(val, sort_keys=True, default=str)
