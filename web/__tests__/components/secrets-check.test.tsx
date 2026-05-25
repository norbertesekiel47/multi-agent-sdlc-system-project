/**
 * Integration test: Verify secrets never appear in rendered HTML.
 *
 * This test checks that API keys and other secrets are never
 * rendered in the HTML output of any page component.
 * (VAL-CROSS-023)
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { TaskSubmitForm } from "@/components/task/task-submit-form";
import { RecentTasks } from "@/components/task/recent-tasks";
import { CostPanel } from "@/components/task/cost-panel";
import { DiffViewer } from "@/components/task/diff-viewer";
import { HitlApproval } from "@/components/task/hitl-approval";
import { screen, waitFor, cleanup } from "@testing-library/react";
import type { TaskDetail } from "@/types/api";

afterEach(() => cleanup());

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

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

// Mock the API client
vi.mock("@/lib/api-client", () => ({
  apiClient: {
    createTask: vi.fn(),
    listTasks: vi.fn().mockResolvedValue({
      tasks: [],
      total: 0,
      limit: 10,
      offset: 0,
    }),
    getTask: vi.fn().mockResolvedValue({
      id: "test-id",
      repo_url: "https://github.com/org/repo",
      issue_number: 1,
      issue_text: "Test issue",
      topology: "hybrid",
      status: "awaiting_hitl",
      total_cost_usd: 0.5,
      total_tokens_in: 100,
      total_tokens_out: 50,
      total_tokens_cached: 20,
      hitl_decision: null,
      pr_url: null,
      started_at: "2026-05-24T10:00:00Z",
      ended_at: null,
    }),
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

// Common secret patterns that should NEVER appear in rendered HTML
const SECRET_PATTERNS = [
  /sk-[a-zA-Z0-9]{20,}/,       // OpenAI-style keys
  /gho_[a-zA-Z0-9]{20,}/,      // GitHub OAuth tokens
  /github_pat_[a-zA-Z0-9]{20,}/, // GitHub PATs
  /hf_[a-zA-Z0-9]{20,}/,       // Hugging Face tokens
  /OPENROUTER_API_KEY/,
  /OPENAI_API_KEY/,
  /GITHUB_PAT/,
  /HUGGINGFACE_TOKEN/,
];

describe("Secrets Hygiene (VAL-CROSS-023)", () => {
  it("no secrets in TaskSubmitForm rendered HTML", () => {
    const { container } = renderWithProviders(<TaskSubmitForm />);
    const html = container.innerHTML;

    for (const pattern of SECRET_PATTERNS) {
      expect(
        pattern.test(html),
        `Found secret pattern ${pattern} in TaskSubmitForm HTML`
      ).toBe(false);
    }
  });

  it("no secrets in RecentTasks rendered HTML", async () => {
    const { container } = renderWithProviders(<RecentTasks />);

    await waitFor(() => {
      expect(screen.getByText(/recent tasks/i)).toBeInTheDocument();
    });

    const html = container.innerHTML;
    for (const pattern of SECRET_PATTERNS) {
      expect(
        pattern.test(html),
        `Found secret pattern ${pattern} in RecentTasks HTML`
      ).toBe(false);
    }
  });

  it("no secrets in CostPanel rendered HTML", () => {
    const task: TaskDetail = {
      id: "test-id",
      repo_url: "https://github.com/org/repo",
      issue_number: 1,
      issue_text: "Test issue",
      topology: "hybrid",
      status: "running",
      total_cost_usd: 0.5,
      total_tokens_in: 100,
      total_tokens_out: 50,
      total_tokens_cached: 20,
      hitl_decision: null,
      pr_url: null,
      started_at: "2026-05-24T10:00:00Z",
      ended_at: null,
    };

    const { container } = renderWithProviders(<CostPanel task={task} />);
    const html = container.innerHTML;

    for (const pattern of SECRET_PATTERNS) {
      expect(
        pattern.test(html),
        `Found secret pattern ${pattern} in CostPanel HTML`
      ).toBe(false);
    }
  });

  it("no secrets in DiffViewer rendered HTML", () => {
    const { container } = renderWithProviders(<DiffViewer />);
    const html = container.innerHTML;

    for (const pattern of SECRET_PATTERNS) {
      expect(
        pattern.test(html),
        `Found secret pattern ${pattern} in DiffViewer HTML`
      ).toBe(false);
    }
  });

  it("no secrets in HitlApproval rendered HTML", async () => {
    const { container } = renderWithProviders(
      <HitlApproval taskId="test-id" />
    );

    await waitFor(() => {
      expect(
        screen.getByText(/human-in-the-loop/i)
      ).toBeInTheDocument();
    });

    const html = container.innerHTML;
    for (const pattern of SECRET_PATTERNS) {
      expect(
        pattern.test(html),
        `Found secret pattern ${pattern} in HitlApproval HTML`
      ).toBe(false);
    }
  });
});
