"""Pydantic request/response models for the FastAPI backend.

All API contracts are defined here.  Models validate input and
structure output for:
  - POST /tasks — create a new task
  - GET /tasks — list tasks with filters
  - GET /tasks/{id} — get task detail
  - WS /events/stream — WebSocket trace event stream
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ── Valid enums (must match episodic models) ────────────────────────

VALID_TOPOLOGIES: frozenset[str] = frozenset(
    {"single_agent", "supervisor_only", "hybrid"}
)


# ── Request models ──────────────────────────────────────────────────


class CreateTaskRequest(BaseModel):
    """Request body for POST /tasks."""

    repo_url: str = Field(..., min_length=1, description="GitHub repository URL")
    issue_number: int = Field(..., gt=0, le=2147483647, description="GitHub issue number")
    issue_text: str = Field(..., min_length=1, description="Issue text / description")
    topology: str = Field(default="hybrid", description="Agent topology")
    auto_start: bool = Field(default=True, description="Start the orchestrator automatically")

    @field_validator("repo_url")
    @classmethod
    def _validate_repo_url(cls, v: str) -> str:
        """Trim whitespace from repo URL."""
        return v.strip()

    @field_validator("topology")
    @classmethod
    def _validate_topology(cls, v: str) -> str:
        """Reject invalid topology values."""
        if v not in VALID_TOPOLOGIES:
            msg = f"Invalid topology {v!r}. Must be one of {sorted(VALID_TOPOLOGIES)}"
            raise ValueError(msg)
        return v


class ListTasksQuery(BaseModel):
    """Query parameters for GET /tasks."""

    repo_url: str | None = None
    status: str | None = None
    outcome: str | None = None
    topology: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


# ── Response models ──────────────────────────────────────────────────


class CreateTaskResponse(BaseModel):
    """Response body for POST /tasks (201 Created)."""

    id: UUID


class TaskDetailResponse(BaseModel):
    """Response body for GET /tasks/{id}."""

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
    agent_costs: dict[str, dict[str, object]] | None = None
    hitl_decision: str | None = None
    pr_url: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    # HITL enrichment fields — populated when status is awaiting_hitl
    pending_diff: str | None = None
    hitl_cause: str | None = None
    hitl_cause_detail: dict[str, object] | None = None
    review_summary: str | None = None
    test_summary: str | None = None
    reject_reason: str | None = None


class TaskListItemResponse(BaseModel):
    """Compact task representation for GET /tasks list."""

    id: UUID
    repo_url: str
    issue_number: int | None = None
    topology: str
    status: str
    total_cost_usd: Decimal | None = None
    started_at: datetime
    ended_at: datetime | None = None


class ListTasksResponse(BaseModel):
    """Response body for GET /tasks."""

    tasks: list[TaskListItemResponse]
    total: int
    limit: int
    offset: int


class HITLDecisionRequest(BaseModel):
    """Request body for POST /tasks/{id}/hitl/decision.

    VAL-HITL-CTRL-010: Invalid decision values (not 'approve' or 'reject')
    trigger FastAPI/Pydantic validation → HTTP 422.
    """

    decision: str = Field(
        ...,
        description="Human decision: 'approve' or 'reject'",
    )
    reason: str | None = Field(default=None, description="Optional reason for the decision")

    @field_validator("decision")
    @classmethod
    def _validate_decision(cls, v: str) -> str:
        """Reject any decision value other than 'approve' or 'reject'."""
        if v not in {"approve", "reject"}:
            msg = f"Invalid decision {v!r}. Must be 'approve' or 'reject'"
            raise ValueError(msg)
        return v


class HITLDecisionResponse(BaseModel):
    """Response body for POST /tasks/{id}/hitl/decision."""

    task_id: UUID
    decision: str
    status: str


class ErrorResponse(BaseModel):
    """Error response — never leaks Python tracebacks (VAL-BACKEND-API-002)."""

    error: str
    detail: str | None = None
