/**
 * Tests for the CostPanel component.
 *
 * Verifies: cost formatting, token display, loading state.
 */

import { describe, it, expect, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { CostPanel } from "@/components/task/cost-panel";
import { screen, cleanup } from "@testing-library/react";

afterEach(() => cleanup());
import type { TaskDetail } from "@/types/api";

const mockTask: TaskDetail = {
  id: "task-123",
  repo_url: "https://github.com/org/repo",
  issue_number: 42,
  issue_text: "Fix the bug",
  topology: "hybrid",
  status: "running",
  total_cost_usd: 0.1234,
  total_tokens_in: 1000,
  total_tokens_out: 500,
  total_tokens_cached: 200,
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

describe("CostPanel", () => {
  it("shows loading state when task is undefined", () => {
    renderWithProviders(<CostPanel task={undefined} />);

    expect(screen.getByText(/loading cost data/i)).toBeInTheDocument();
  });

  it("displays cost values formatted correctly", () => {
    renderWithProviders(
      <CostPanel task={mockTask} events={[]} isTerminal={false} />
    );

    // $0.1234 should appear somewhere in the component
    expect(screen.getByTestId("total-cost")).toHaveTextContent("$0.1234");
    expect(screen.getByTestId("total-tokens-in")).toHaveTextContent("1,000");
    expect(screen.getByTestId("total-tokens-out")).toHaveTextContent("500");
    expect(screen.getByTestId("total-cached-tokens")).toHaveTextContent("200");
  });

  it("shows dash for null cost values when no events provide data", () => {
    const nullCostTask = {
      ...mockTask,
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
});
