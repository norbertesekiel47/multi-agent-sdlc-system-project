/**
 * Tests for the TaskDetailContent component (inner component of TaskDetailPage).
 *
 * Verifies: topology display, HITL link, PR link, topology transition visualization.
 * The full page uses React `use()` with async params which doesn't play well
 * with test environments, so we test the inner content component directly.
 */

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { TaskDetailContent } from "@/components/task/task-detail-content";
import { screen, waitFor, cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// Mock next/link
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

// Mock the useTraceStream hook
vi.mock("@/hooks/use-trace-stream", () => ({
  useTraceStream: () => ({
    events: [],
    status: "connected",
    error: null,
  }),
}));

// Mock the useTask hook
vi.mock("@/hooks/use-tasks", () => ({
  useTask: () => ({
    data: null,
    isLoading: false,
    error: null,
  }),
  useTasks: () => ({
    data: null,
    isLoading: false,
    error: null,
  }),
  useCreateTask: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  useHitlDecision: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  taskKeys: {
    all: ["tasks"],
    detail: (id: string) => ["tasks", "detail", id],
  },
}));

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

describe("TaskDetailContent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("displays task topology in header", () => {
    renderWithProviders(
      <TaskDetailContent id="task-123" task={mockTask} />
    );

    // "hybrid" should appear in the header metadata
    expect(screen.getAllByText(/hybrid/i).length).toBeGreaterThanOrEqual(1);
  });

  it("shows status badge", () => {
    renderWithProviders(
      <TaskDetailContent id="task-123" task={mockTask} />
    );

    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("shows HITL review link when task is awaiting_hitl", () => {
    renderWithProviders(
      <TaskDetailContent
        id="task-123"
        task={{ ...mockTask, status: "awaiting_hitl" }}
      />
    );

    expect(screen.getByText(/review pending decision/i)).toBeInTheDocument();
  });

  it("shows PR link when task has pr_url", () => {
    renderWithProviders(
      <TaskDetailContent
        id="task-123"
        task={{
          ...mockTask,
          status: "completed",
          pr_url: "https://github.com/org/repo/pull/1",
        }}
      />
    );

    expect(screen.getByText(/view pull request/i)).toBeInTheDocument();
  });

  it("shows topology transition visualization", () => {
    renderWithProviders(
      <TaskDetailContent id="task-123" task={mockTask} />
    );

    // The Agent Flow section should be rendered
    expect(screen.getByText(/agent flow/i)).toBeInTheDocument();
  });

  it("shows single agent flow for single_agent topology", () => {
    renderWithProviders(
      <TaskDetailContent
        id="task-123"
        task={{ ...mockTask, topology: "single_agent" }}
      />
    );

    expect(screen.getByText(/single agent/i)).toBeInTheDocument();
  });

  it("shows Coder ⇄ Reviewer back-edge for hybrid topology", () => {
    renderWithProviders(
      <TaskDetailContent id="task-123" task={mockTask} />
    );

    expect(screen.getByText(/coder.*reviewer/i)).toBeInTheDocument();
  });

  it("does not show HITL link for running task", () => {
    renderWithProviders(
      <TaskDetailContent id="task-123" task={mockTask} />
    );

    expect(screen.queryByText(/review pending decision/i)).not.toBeInTheDocument();
  });
});
