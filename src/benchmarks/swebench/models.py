"""Pydantic models for SWE-bench instances and results.

These typed schemas define the data flowing through the harness.
VAL-SWE-BENCH-001 requires a typed SweBenchInstance.
VAL-SWE-BENCH-005 requires a typed SweBenchResult.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SweBenchInstance(BaseModel):
    """A single SWE-bench-Lite instance fetched from HuggingFace.

    Fields map to the princeton-nlp/SWE-bench_Lite test split columns.
    All required fields are non-null per VAL-SWE-BENCH-001.

    The SWE-bench dataset uses various column names; this model
    normalizes them to consistent Python identifiers.
    """

    instance_id: str = Field(
        ..., alias="instance_id", min_length=1,
        description="Unique instance identifier (e.g. 'django__django-12345')",
    )
    repo: str = Field(
        ..., alias="repo", min_length=1,
        description="Repository in 'owner/repo' format",
    )
    base_commit: str = Field(
        ..., alias="base_commit", min_length=1,
        description="Git commit SHA the instance is based on",
    )
    problem_statement: str = Field(
        ..., alias="problem_statement", min_length=1,
        description="The issue/problem description text",
    )
    hints_text: str = Field(
        default="", alias="hints_text",
        description="Optional hints for solving the issue",
    )
    created_at: str = Field(
        default="", alias="created_at",
        description="ISO timestamp when the instance was created",
    )
    version: str = Field(
        default="", alias="version",
        description="Package version string",
    )
    FAIL_TO_PASS: list[str] = Field(
        default_factory=list, alias="FAIL_TO_PASS",
        description="Test names that should pass after the fix",
    )
    PASS_TO_PASS: list[str] = Field(
        default_factory=list, alias="PASS_TO_PASS",
        description="Test names that should still pass after the fix",
    )
    test_patch: str = Field(
        default="", alias="test_patch",
        description="Unified diff for the test file changes",
    )
    patch: str = Field(
        default="", alias="patch",
        description="The gold-standard patch (ground truth)",
    )

    model_config = {"populate_by_name": True}

    @field_validator("instance_id", "repo", "base_commit", "problem_statement")
    @classmethod
    def _required_fields_non_empty(cls, v: str) -> str:
        """Required fields must be non-empty strings (VAL-SWE-BENCH-001)."""
        if not v or not v.strip():
            msg = "Required field must be non-empty"
            raise ValueError(msg)
        return v


class SweBenchResult(BaseModel):
    """Evaluation result for a single SWE-bench instance.

    Produced by the evaluator wrapper (VAL-SWE-BENCH-004)
    and parsed into typed form (VAL-SWE-BENCH-005).
    """

    __test__ = False  # Prevent pytest collection

    instance_id: str = Field(
        ..., min_length=1,
        description="Instance identifier matching SweBenchInstance.instance_id",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the patch resolved the instance",
    )
    pass_count: int = Field(
        default=0, ge=0,
        description="Number of tests passing after the patch",
    )
    fail_count: int = Field(
        default=0, ge=0,
        description="Number of tests failing after the patch",
    )
    error: str | None = Field(
        default=None,
        description="Error message if evaluation failed, None on success",
    )
    model_patch: str = Field(
        default="",
        description="The model-generated patch that was evaluated",
    )
    tests_status: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-test status dict from the evaluator",
    )


class RunConfig(BaseModel):
    """Configuration for a SWE-bench benchmark run.

    Controls the slice size, topology, temperature, and other
    run parameters.  Used by the CLI entry point and the runner.
    """

    slice_size: int = Field(
        default=30, ge=1, le=300,
        description="Number of instances to run (default 30)",
    )
    topology: str = Field(
        default="supervisor_only",
        description="Orchestrator topology: single_agent, supervisor_only, or hybrid",
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="LLM temperature (0 for benchmark runs, 0.2 for normal)",
    )
    runs_per_cell: int = Field(
        default=1, ge=1, le=10,
        description="Number of repeated runs per (topology, instance) cell (N=3 for final)",
    )
    max_cost_per_task_usd: float = Field(
        default=2.00, ge=0.01,
        description="Maximum cost per task in USD",
    )
    output_dir: str = Field(
        default="benchmarks/results",
        description="Directory to write results",
    )
    instance_ids: list[str] | None = Field(
        default=None,
        description="Specific instance IDs to run (overrides slice_size)",
    )
