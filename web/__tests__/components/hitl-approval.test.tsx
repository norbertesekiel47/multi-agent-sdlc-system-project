/**
 * Tests for the HitlApproval component.
 *
 * Verifies: diff rendering, approve/reject buttons,
 * 409 conflict handling, XSS safety, state management.
 */

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { HitlApproval } from "@/components/task/hitl-approval";
import { screen, waitFor, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

afterEach(() => cleanup());

// Mock the API client
vi.mock("@/lib/api-client", () => ({
  apiClient: {
    createTask: vi.fn(),
    listTasks: vi.fn(),
    getTask: vi.fn(),
    hitlDecision: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    body: unknown;
    constructor(status: number, body: unknown) {
      super(`API error ${status}`);
      this.name = "ApiError";
      this.status = status;
      this.body = body;
    }
  },
}));

import { apiClient } from "@/lib/api-client";

const awaitingHitlTask = {
  id: "task-123",
  repo_url: "https://github.com/org/repo",
  issue_number: 42,
  issue_text: "Fix the bug",
  topology: "hybrid",
  status: "awaiting_hitl",
  total_cost_usd: 0.5,
  total_tokens_in: 1000,
  total_tokens_out: 500,
  total_tokens_cached: 200,
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

describe("HitlApproval", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders diff viewer and approve/reject buttons for awaiting_hitl task", async () => {
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByText(/human-in-the-loop review/i)).toBeInTheDocument();
    });

    // Use role-based selectors for buttons
    expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^reject$/i })).toBeInTheDocument();
  });

  it("shows awaiting decision badge for awaiting_hitl task", async () => {
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByText(/awaiting decision/i)).toBeInTheDocument();
    });
  });

  it("shows no pending decision message for running task", async () => {
    vi.mocked(apiClient.getTask).mockResolvedValue({
      ...awaitingHitlTask,
      status: "running",
    } as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByText(/no pending decision/i)).toBeInTheDocument();
    });
  });

  it("shows decision already made for approved task", async () => {
    vi.mocked(apiClient.getTask).mockResolvedValue({
      ...awaitingHitlTask,
      status: "completed",
      hitl_decision: "approve",
      pr_url: "https://github.com/org/repo/pull/1",
    } as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByText(/approved/i)).toBeInTheDocument();
    });
  });

  it("approve button calls API with decision=approve", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);
    vi.mocked(apiClient.hitlDecision).mockResolvedValue({
      task_id: "task-123",
      decision: "approve",
      status: "running",
    } as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /approve/i }));

    await waitFor(() => {
      expect(apiClient.hitlDecision).toHaveBeenCalledWith("task-123", {
        decision: "approve",
      });
    });
  });

  it("reject with reason sends reason in body", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);
    vi.mocked(apiClient.hitlDecision).mockResolvedValue({
      task_id: "task-123",
      decision: "reject",
      status: "rejected",
    } as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^reject$/i })).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText(/reject reason/i), "Code quality issues");
    await user.click(screen.getByRole("button", { name: /^reject$/i }));

    await waitFor(() => {
      expect(apiClient.hitlDecision).toHaveBeenCalledWith("task-123", {
        decision: "reject",
        reason: "Code quality issues",
      });
    });
  });

  it("surfaces 409 conflict error on second decision", async () => {
    const user = userEvent.setup();
    const { ApiError } = await import("@/lib/api-client");
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);
    vi.mocked(apiClient.hitlDecision).mockRejectedValue(
      new ApiError(409, { error: "decision_already_made" }) as never
    );

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /approve/i }));

    await waitFor(() => {
      expect(screen.getByText(/decision has already been made/i)).toBeInTheDocument();
    });
  });

  it("reject reason with HTML tags is escaped (no XSS)", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^reject$/i })).toBeInTheDocument();
    });

    const maliciousReason = '<script>alert("xss")</script>';
    await user.type(screen.getByLabelText(/reject reason/i), maliciousReason);

    // The textarea should contain the text literally
    const textarea = screen.getByLabelText(/reject reason/i) as HTMLTextAreaElement;
    expect(textarea.value).toBe(maliciousReason);

    // No script element should be in the document
    expect(document.querySelector("script:not([src])")).toBeNull();
  });

  it("approve and reject buttons are disabled during in-flight decision", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.getTask).mockResolvedValue(awaitingHitlTask as never);
    // Make the API call hang
    vi.mocked(apiClient.hitlDecision).mockReturnValue(new Promise(() => {}) as never);

    renderWithProviders(<HitlApproval taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    });

    const approveBtn = screen.getByRole("button", { name: /approve/i });
    const rejectBtn = screen.getByRole("button", { name: /^reject$/i });

    await user.click(approveBtn);

    await waitFor(() => {
      expect(approveBtn).toBeDisabled();
      expect(rejectBtn).toBeDisabled();
    });
  });
});
