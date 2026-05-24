"""SQL DDL for the episodic memory store.

Creates the four tables specified in architecture.md §2.6:
  - tasks
  - decisions
  - outcomes
  - repo_facts

Status and outcome enums are enforced via CHECK constraints.
FK constraints use ON DELETE CASCADE for decisions/outcomes.
repo_facts has a UNIQUE constraint on (repo_url, fact_kind).
"""

from __future__ import annotations

EPISODIC_SCHEMA_SQL: str = """\
-- ────────────────────────────────────────────────────────────────
-- Episodic memory store — architecture.md §2.6
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tasks (
  id              UUID PRIMARY KEY,
  repo_url        TEXT NOT NULL,
  issue_number    INTEGER,
  issue_text      TEXT NOT NULL,
  topology        TEXT NOT NULL,
  status          TEXT NOT NULL,
  total_cost_usd  NUMERIC(10,4),
  total_tokens_in   INTEGER,
  total_tokens_out  INTEGER,
  total_tokens_cached INTEGER,
  hitl_decision   TEXT,
  pr_url          TEXT,
  started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at        TIMESTAMPTZ,

  CONSTRAINT tasks_status_check
    CHECK (status IN ('running','awaiting_hitl','approved','rejected','completed','failed'))
);

CREATE INDEX IF NOT EXISTS tasks_repo_url_idx ON tasks(repo_url);
CREATE INDEX IF NOT EXISTS tasks_status_idx ON tasks(status);

CREATE TABLE IF NOT EXISTS decisions (
  id              UUID PRIMARY KEY,
  task_id         UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  agent           TEXT NOT NULL,
  step_index      INTEGER NOT NULL,
  decision_type   TEXT NOT NULL,
  decision_data   JSONB NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS decisions_task_id_idx ON decisions(task_id);
CREATE INDEX IF NOT EXISTS decisions_agent_idx ON decisions(agent);
CREATE INDEX IF NOT EXISTS decisions_decision_type_idx ON decisions(decision_type);

CREATE TABLE IF NOT EXISTS outcomes (
  id              UUID PRIMARY KEY,
  task_id         UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  outcome         TEXT NOT NULL,
  detail          JSONB,
  recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT outcomes_outcome_check
    CHECK (outcome IN (
      'success',
      'pr_opened',
      'hitl_rejected',
      'retry_budget_exhausted',
      'loop_detected',
      'uncertainty_escalation',
      'guardrail_block',
      'cost_budget_exhausted',
      'sandbox_failure'
    ))
);

CREATE INDEX IF NOT EXISTS outcomes_task_id_idx ON outcomes(task_id);
CREATE INDEX IF NOT EXISTS outcomes_outcome_idx ON outcomes(outcome);

CREATE TABLE IF NOT EXISTS repo_facts (
  id              UUID PRIMARY KEY,
  repo_url        TEXT NOT NULL,
  fact_kind       TEXT NOT NULL,
  fact_value      JSONB NOT NULL,
  observed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT repo_facts_repo_url_fact_kind_uniq
    UNIQUE (repo_url, fact_kind)
);

CREATE INDEX IF NOT EXISTS repo_facts_repo_url_idx ON repo_facts(repo_url);
"""
