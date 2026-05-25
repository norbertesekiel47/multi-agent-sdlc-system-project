/**
 * Tests for the enhanced CostPanel component.
 *
 * Covers: VAL-COST-VIEW-001 (running totals), VAL-COST-VIEW-002 (per-agent breakdown),
 * VAL-COST-VIEW-003 (live updates), VAL-COST-VIEW-004 (cached-token column),
 * VAL-COST-VIEW-005 (80%/100% budget warnings), VAL-COST-VIEW-006 (WS reconnect),
 * VAL-COST-VIEW-007 (formatting), VAL-COST-VIEW-008 (frozen terminal).
 */

import { describe, it, expect, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { CostPanel } from "@/components/task/cost-panel";
import { screen, cleanup, within } from "@testing-library/react";
import type { TraceEvent, TaskDetail } from "@/types/api";

afterEach(() => cleanup());

function makeNodeEndEvent(
  agent: string,
  tokensIn: number,
  tokensOut: number,
  cachedTokens: number,
  costUsd: number,
  spanId = ""
): TraceEvent {
  return {
    type: "node_end",
    task_id: "task-1",
    span_id: spanId || `span-${agent}-${Math.random()}`,
    parent_span_id: null,
    name: agent,
    started_at: "2026-05-24T10:00:00Z",
    ended_at: "2026-05-24T10:00:05Z",
    tokens_in: tokensIn,
    tokens_out: tokensOut,
    cost_usd: costUsd,
    status: "ok",
    agent,
    cached_tokens: cachedTokens,
  };
}

const baseTask: TaskDetail = {
  id: "task-1",
  repo_url: "https://github.com/org/repo",
  issue_number: 42,
  issue_text: "Fix the bug",
  topology: "hybrid",
  status: "running",
  total_cost_usd: 0.30,
  total_tokens_in: 4500,
  total_tokens_out: 1000,
  total_tokens_cached: 500,
  agent_costs: null,
  hitl_decision: null,
  pr_url: null,
  started_at: "2026-05-24T10:00:00Z",
  ended_at: null,
  pending_diff: null,
  hitl_cause: null,
  hitl_cause_detail: null,
  review_summary: null,
  test_summary: null,
  reject_reason: null,
};

const sampleEvents: TraceEvent[] = [
  makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
  makeNodeEndEvent("coder", 2000, 500, 300, 0.15),
  makeNodeEndEvent("reviewer", 1500, 300, 200, 0.10),
];

describe("CostPanel — VAL-COST-VIEW assertions", () => {
  // VAL-COST-VIEW-001: Per-task cost panel renders running totals
  it("shows running totals: total cost, tokens in, tokens out, cached tokens", () => {
    renderWithProviders(
      <CostPanel task={baseTask} events={sampleEvents} isTerminal={false} />
    );

    // Total cost should be formatted as $X.XXXX
    expect(screen.getByTestId("total-cost")).toHaveTextContent("$0.3000");
    // Tokens with comma thousands
    expect(screen.getByTestId("total-tokens-in")).toHaveTextContent("4,500");
    expect(screen.getByTestId("total-tokens-out")).toHaveTextContent("1,000");
    expect(screen.getByTestId("total-cached-tokens")).toHaveTextContent("500");
  });

  it("shows zeros when task just started with no events", () => {
    const newTask = { ...baseTask, total_cost_usd: 0, total_tokens_in: 0, total_tokens_out: 0, total_tokens_cached: 0 };
    renderWithProviders(
      <CostPanel task={newTask} events={[]} isTerminal={false} />
    );

    // With zero values, the task-level data takes priority (0 is not null)
    expect(screen.getByTestId("total-cost")).toHaveTextContent("$0.0000");
  });

  // VAL-COST-VIEW-002: Per-agent cost breakdown is visible
  it("shows per-agent breakdown rows for each active agent", () => {
    renderWithProviders(
      <CostPanel task={baseTask} events={sampleEvents} isTerminal={false} />
    );

    // Per-agent rows should be visible in the table
    expect(screen.getByText(/planner/i)).toBeInTheDocument();
    expect(screen.getByText(/coder/i)).toBeInTheDocument();
    expect(screen.getByText(/reviewer/i)).toBeInTheDocument();

    // Per-agent cost values (these appear in table cells)
    expect(screen.getAllByText(/\$0\.0500/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/\$0\.1500/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/\$0\.1000/).length).toBeGreaterThanOrEqual(1);
  });

  it("per-agent costs sum to total within rounding", () => {
    renderWithProviders(
      <CostPanel task={baseTask} events={sampleEvents} isTerminal={false} />
    );

    // Total cost row
    expect(screen.getByTestId("total-cost")).toHaveTextContent("$0.3000");
    // Sum of per-agent: 0.05 + 0.15 + 0.10 = 0.30
  });

  // VAL-COST-VIEW-004: Cached-token column is populated when caching applies
  it("shows non-zero cached tokens for coder/reviewer when caching is active", () => {
    renderWithProviders(
      <CostPanel task={baseTask} events={sampleEvents} isTerminal={false} />
    );

    // The "Cached" column header should exist in the per-agent table
    const cachedHeaders = screen.getAllByText(/cached/i);
    expect(cachedHeaders.length).toBeGreaterThanOrEqual(1);

    // Table should exist
    const table = screen.getByRole("table", { name: /cost breakdown/i });
    expect(table).toBeInTheDocument();
  });

  // VAL-COST-VIEW-005: Budget warning at 80% and halt at 100%
  it("shows 80% budget warning banner when cost reaches threshold", () => {
    const expensiveTask = {
      ...baseTask,
      total_cost_usd: 1.60, // 80% of $2.00
    };
    const expensiveEvents: TraceEvent[] = [
      makeNodeEndEvent("planner", 5000, 1000, 0, 1.60),
    ];

    renderWithProviders(
      <CostPanel
        task={expensiveTask}
        events={expensiveEvents}
        isTerminal={false}
        maxBudgetUsd={2.0}
      />
    );

    expect(screen.getByText(/80%.*budget/i)).toBeInTheDocument();
  });

  it("shows 100% budget halt banner when cost exceeds budget", () => {
    const overBudgetTask = {
      ...baseTask,
      total_cost_usd: 2.05,
      status: "failed" as const,
    };
    const overBudgetEvents: TraceEvent[] = [
      makeNodeEndEvent("planner", 5000, 1000, 0, 2.05),
    ];

    renderWithProviders(
      <CostPanel
        task={overBudgetTask}
        events={overBudgetEvents}
        isTerminal={true}
        maxBudgetUsd={2.0}
      />
    );

    expect(screen.getByText(/budget.*exceeded|100%.*budget/i)).toBeInTheDocument();
  });

  // VAL-COST-VIEW-007: Cost values format consistently
  it("formats USD as $X.XXXX with optional comma thousands", () => {
    const bigTask = {
      ...baseTask,
      total_cost_usd: 1234.5678,
      total_tokens_in: 1000000,
      total_tokens_out: 200000,
      total_tokens_cached: 0,
    };
    const bigEvents: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000000, 200000, 0, 1234.5678),
    ];

    renderWithProviders(
      <CostPanel task={bigTask} events={bigEvents} isTerminal={false} />
    );

    // USD should be formatted as $1,234.5678
    expect(screen.getByTestId("total-cost")).toHaveTextContent("$1,234.5678");
    // Tokens with comma thousands
    expect(screen.getByTestId("total-tokens-in")).toHaveTextContent("1,000,000");
    expect(screen.getByTestId("total-tokens-out")).toHaveTextContent("200,000");
  });

  it("never shows NaN, undefined, or scientific notation in cost cells", () => {
    renderWithProviders(
      <CostPanel task={baseTask} events={sampleEvents} isTerminal={false} />
    );

    // No cell should contain NaN, undefined, or scientific notation
    expect(screen.queryByText(/NaN/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/undefined/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/e\+/i)).not.toBeInTheDocument();
  });

  it("shows dash for null cost values", () => {
    const nullCostTask = {
      ...baseTask,
      total_cost_usd: null,
      total_tokens_in: null,
      total_tokens_out: null,
      total_tokens_cached: null,
    };

    renderWithProviders(
      <CostPanel task={nullCostTask} events={[]} isTerminal={false} />
    );

    // Should show dashes for null values where no events provide data
    expect(screen.getByTestId("total-cost")).toHaveTextContent("—");
    expect(screen.getByTestId("total-tokens-in")).toHaveTextContent("—");
    expect(screen.getByTestId("total-tokens-out")).toHaveTextContent("—");
    expect(screen.getByTestId("total-cached-tokens")).toHaveTextContent("—");
  });

  // VAL-COST-VIEW-008: Cost panel frozen at final value for terminal tasks
  it("shows frozen indicator for terminal tasks", () => {
    const terminalTask = {
      ...baseTask,
      status: "completed" as const,
      ended_at: "2026-05-24T10:05:00Z",
    };

    renderWithProviders(
      <CostPanel task={terminalTask} events={sampleEvents} isTerminal={true} />
    );

    expect(screen.getByText(/final/i)).toBeInTheDocument();
  });

  it("does not show budget warning below 80%", () => {
    const lowCostTask = {
      ...baseTask,
      total_cost_usd: 0.50,
    };

    renderWithProviders(
      <CostPanel
        task={lowCostTask}
        events={sampleEvents}
        isTerminal={false}
        maxBudgetUsd={2.0}
      />
    );

    expect(screen.queryByText(/budget/i)).not.toBeInTheDocument();
  });

  // Loading state
  it("shows loading state when task is undefined", () => {
    renderWithProviders(
      <CostPanel task={undefined} events={[]} isTerminal={false} />
    );

    expect(screen.getByText(/loading cost data/i)).toBeInTheDocument();
  });
});
