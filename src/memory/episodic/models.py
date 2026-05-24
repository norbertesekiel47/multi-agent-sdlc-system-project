"""Pydantic models for the episodic memory store.

Each model maps 1:1 to a database row in the episodic schema (§2.6).
Enums are enforced at both the DB CHECK-constraint level and at the
application level via Pydantic validators.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, field_validator

# ── Status enum ──────────────────────────────────────────────────────────────

VALID_STATUSES: frozenset[str] = frozenset(
    {"running", "awaiting_hitl", "approved", "rejected", "completed", "failed"}
)


# ── Outcome enum ─────────────────────────────────────────────────────────────

VALID_OUTCOMES: frozenset[str] = frozenset(
    {
        "success",
        "pr_opened",
        "hitl_rejected",
        "retry_budget_exhausted",
        "loop_detected",
        "uncertainty_escalation",
        "guardrail_block",
        "cost_budget_exhausted",
        "sandbox_failure",
    }
)


class TaskRow(BaseModel):
    """Maps to the ``tasks`` table."""

    id: UUID
    repo_url: str
    issue_number: int | None = None
    issue_text: str
    topology: str
    status: str
    total_cost_usd: Decimal | None = None
    total_tokens_in: int | None = None
    total_tokens_out: int | None = None
    total_tokens_cached: int | None = None
    hitl_decision: str | None = None
    pr_url: str | None = None
    started_at: datetime
    ended_at: datetime | None = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            msg = f"Invalid status {v!r}. Must be one of {sorted(VALID_STATUSES)}"
            raise ValueError(msg)
        return v


class DecisionRow(BaseModel):
    """Maps to the ``decisions`` table."""

    id: UUID
    task_id: UUID
    agent: str
    step_index: int
    decision_type: str
    decision_data: dict[str, Any]
    created_at: datetime


class OutcomeRow(BaseModel):
    """Maps to the ``outcomes`` table."""

    id: UUID
    task_id: UUID
    outcome: str
    detail: dict[str, Any] | None = None
    recorded_at: datetime

    @field_validator("outcome")
    @classmethod
    def _validate_outcome(cls, v: str) -> str:
        if v not in VALID_OUTCOMES:
            msg = f"Invalid outcome {v!r}. Must be one of {sorted(VALID_OUTCOMES)}"
            raise ValueError(msg)
        return v


class RepoFactRow(BaseModel):
    """Maps to the ``repo_facts`` table."""

    id: UUID
    repo_url: str
    fact_kind: str
    fact_value: dict[str, Any]
    observed_at: datetime


class CreateTaskParams(BaseModel):
    """Parameters for creating a new task row."""

    repo_url: str
    issue_number: int | None = None
    issue_text: str
    topology: str
    status: str = "running"

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            msg = f"Invalid status {v!r}. Must be one of {sorted(VALID_STATUSES)}"
            raise ValueError(msg)
        return v


class CreateDecisionParams(BaseModel):
    """Parameters for creating a new decision row."""

    task_id: UUID
    agent: str
    step_index: int
    decision_type: str
    decision_data: dict[str, Any]


class CreateOutcomeParams(BaseModel):
    """Parameters for creating a new outcome row."""

    task_id: UUID
    outcome: str
    detail: dict[str, Any] | None = None

    @field_validator("outcome")
    @classmethod
    def _validate_outcome(cls, v: str) -> str:
        if v not in VALID_OUTCOMES:
            msg = f"Invalid outcome {v!r}. Must be one of {sorted(VALID_OUTCOMES)}"
            raise ValueError(msg)
        return v


class UpsertRepoFactParams(BaseModel):
    """Parameters for upserting a repo_fact row."""

    repo_url: str
    fact_kind: str
    fact_value: dict[str, Any]
