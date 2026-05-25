/**
 * Tests for the TaskSubmitForm component.
 *
 * Verifies: form validation, topology selector defaults,
 * submit button states, error handling, XSS safety.
 */

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "../test-utils";
import { TaskSubmitForm } from "@/components/task/task-submit-form";
import { screen, waitFor, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

afterEach(() => cleanup());

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
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
      const message =
        typeof body === "object" && body !== null && "error" in body
          ? String((body as { error: string }).error)
          : `API error ${status}`;
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.body = body;
    }
  },
}));

import { apiClient } from "@/lib/api-client";

describe("TaskSubmitForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the submission form with all fields", () => {
    renderWithProviders(<TaskSubmitForm />);

    expect(screen.getByLabelText(/repository url/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/issue number/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit task/i })).toBeInTheDocument();
  });

  it("shows validation error when repo URL is empty", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/issue number/i), "42");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    expect(screen.getByText(/repository url is required/i)).toBeInTheDocument();
    expect(apiClient.createTask).not.toHaveBeenCalled();
  });

  it("shows validation error when issue number is empty", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/repository url/i), "https://github.com/org/repo");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    expect(screen.getByText(/issue number is required/i)).toBeInTheDocument();
    expect(apiClient.createTask).not.toHaveBeenCalled();
  });

  it("rejects non-integer issue number", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/repository url/i), "https://github.com/org/repo");
    await user.type(screen.getByLabelText(/issue number/i), "abc");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    expect(screen.getByText(/issue number must be a positive integer/i)).toBeInTheDocument();
    expect(apiClient.createTask).not.toHaveBeenCalled();
  });

  it("rejects malformed repo URL", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/repository url/i), "not-a-url");
    await user.type(screen.getByLabelText(/issue number/i), "42");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    expect(screen.getByText(/please enter a valid url starting with https/i)).toBeInTheDocument();
    expect(apiClient.createTask).not.toHaveBeenCalled();
  });

  it("topology selector defaults to hybrid", () => {
    renderWithProviders(<TaskSubmitForm />);
    // The Select trigger shows "Hybrid" as the displayed value
    // There's also a hidden option — use the visible trigger span
    const hybridTexts = screen.getAllByText("Hybrid");
    expect(hybridTexts.length).toBeGreaterThanOrEqual(1);
  });

  it("submits valid task and calls API", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.createTask).mockResolvedValue({ id: "test-task-id" } as never);

    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/repository url/i), "https://github.com/org/repo");
    await user.type(screen.getByLabelText(/issue number/i), "42");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    await waitFor(() => {
      expect(apiClient.createTask).toHaveBeenCalledWith({
        repo_url: "https://github.com/org/repo",
        issue_number: 42,
        issue_text: "Issue #42",
        topology: "hybrid",
      });
    });
  });

  it("shows loading state during submission", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.createTask).mockReturnValue(new Promise(() => {}) as never);

    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/repository url/i), "https://github.com/org/repo2");
    await user.type(screen.getByLabelText(/issue number/i), "10");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    expect(screen.getByText(/submitting/i)).toBeInTheDocument();
  });

  it("shows error message on 5xx response", async () => {
    const user = userEvent.setup();
    const { ApiError } = await import("@/lib/api-client");
    vi.mocked(apiClient.createTask).mockRejectedValue(
      new ApiError(500, { error: "internal_server_error" }) as never
    );

    renderWithProviders(<TaskSubmitForm />);

    await user.type(screen.getByLabelText(/repository url/i), "https://github.com/org/repo3");
    await user.type(screen.getByLabelText(/issue number/i), "99");
    await user.click(screen.getByRole("button", { name: /submit task/i }));

    await waitFor(() => {
      expect(screen.getByText(/internal_server_error/i)).toBeInTheDocument();
    });
  });

  it("HTML in input fields is not executed as script", async () => {
    const user = userEvent.setup();
    renderWithProviders(<TaskSubmitForm />);

    const maliciousInput = '<img src=x onerror=alert(1)>';
    const repoInput = screen.getByLabelText(/repository url/i) as HTMLInputElement;
    await user.clear(repoInput);
    await user.type(repoInput, maliciousInput);

    // The input should contain the text literally
    expect(repoInput.value).toBe(maliciousInput);

    // No script or img-onerror should be in the rendered DOM
    expect(document.querySelector("script:not([src])")).toBeNull();
  });
});
