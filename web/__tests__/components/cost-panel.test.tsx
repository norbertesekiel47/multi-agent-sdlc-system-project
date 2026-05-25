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
  hitl_decision: null,
  pr_url: null,
  started_at: "2026-05-24T10:00:00Z",
  ended_at: null,
};

describe("CostPanel", () => {
  it("shows loading state when task is undefined", () => {
    renderWithProviders(<CostPanel task={undefined} />);

    expect(screen.getByText(/loading cost data/i)).toBeInTheDocument();
  });

  it("displays cost values formatted correctly", () => {
    renderWithProviders(<CostPanel task={mockTask} />);

    expect(screen.getByText("$0.1234")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
    expect(screen.getByText("500")).toBeInTheDocument();
    expect(screen.getByText("200")).toBeInTheDocument();
  });

  it("shows dash for null cost values", () => {
    const nullCostTask = {
      ...mockTask,
      total_cost_usd: null,
      total_tokens_in: null,
      total_tokens_out: null,
      total_tokens_cached: null,
    };

    renderWithProviders(<CostPanel task={nullCostTask} />);

    // Should show dashes for null values
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(4);
  });
});
