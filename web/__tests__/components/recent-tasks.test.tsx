/**
 * Tests for the RecentTasks component.
 *
 * Verifies: empty state rendering, loading state, error state,
 * task list rendering.
 */

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { RecentTasks } from "@/components/task/recent-tasks";
import { screen, waitFor, cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    React.createElement("a", { href }, children),
}));

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
      this.status = status;
      this.body = body;
    }
  },
}));

import { apiClient } from "@/lib/api-client";

describe("RecentTasks", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders empty state when no tasks exist", async () => {
    vi.mocked(apiClient.listTasks).mockResolvedValue({
      tasks: [],
      total: 0,
      limit: 10,
      offset: 0,
    } as never);

    renderWithProviders(<RecentTasks />);

    await waitFor(() => {
      expect(
        screen.getByText(/no tasks yet/i)
      ).toBeInTheDocument();
    });
  });

  it("renders loading state", () => {
    vi.mocked(apiClient.listTasks).mockReturnValue(
      new Promise(() => {}) as never
    );

    renderWithProviders(<RecentTasks />);

    // Skeleton elements should be present
    expect(screen.getByText(/recent tasks/i)).toBeInTheDocument();
  });

  it("renders error state", async () => {
    const { ApiError } = await import("@/lib/api-client");
    vi.mocked(apiClient.listTasks).mockRejectedValue(
      new ApiError(500, { error: "server_error" }) as never
    );

    renderWithProviders(<RecentTasks />);

    await waitFor(() => {
      expect(
        screen.getByText(/unable to load tasks/i)
      ).toBeInTheDocument();
    });
  });

  it("renders task list with correct data", async () => {
    const mockTasks = {
      tasks: [
        {
          id: "abc-123",
          repo_url: "https://github.com/org/repo",
          issue_number: 42,
          topology: "hybrid",
          status: "running",
          total_cost_usd: 0.1234,
          started_at: "2026-05-24T10:00:00Z",
          ended_at: null,
        },
      ],
      total: 1,
      limit: 10,
      offset: 0,
    };

    vi.mocked(apiClient.listTasks).mockResolvedValue(mockTasks as never);

    renderWithProviders(<RecentTasks />);

    await waitFor(() => {
      expect(screen.getByText("org/repo")).toBeInTheDocument();
      expect(screen.getByText("#42")).toBeInTheDocument();
      expect(screen.getByText("running")).toBeInTheDocument();
    });
  });
});
