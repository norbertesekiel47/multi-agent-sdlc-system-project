/**
 * Shared API client for communicating with the FastAPI backend.
 *
 * All fetch calls go through this utility. The base URL is
 * configurable via NEXT_PUBLIC_API_URL env var (default: http://localhost:3100).
 * The frontend NEVER calls external services directly.
 */

import type {
  CreateTaskRequest,
  CreateTaskResponse,
  HITLDecisionRequest,
  HITLDecisionResponse,
  ListTasksResponse,
  TaskDetail,
} from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3100";

class ApiError extends Error {
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
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    throw new ApiError(response.status, body);
  }

  return response.json() as Promise<T>;
}

export const apiClient = {
  /** Create a new task. POST /tasks */
  createTask: (data: CreateTaskRequest): Promise<CreateTaskResponse> =>
    request<CreateTaskResponse>("/tasks", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  /** List tasks with optional filters. GET /tasks */
  listTasks: (params?: {
    repo_url?: string;
    status?: string;
    outcome?: string;
    topology?: string;
    limit?: number;
    offset?: number;
  }): Promise<ListTasksResponse> => {
    const searchParams = new URLSearchParams();
    if (params) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
          searchParams.set(key, String(value));
        }
      }
    }
    const qs = searchParams.toString();
    return request<ListTasksResponse>(`/tasks${qs ? `?${qs}` : ""}`);
  },

  /** Get a single task by ID. GET /tasks/{id} */
  getTask: (id: string): Promise<TaskDetail> =>
    request<TaskDetail>(`/tasks/${id}`),

  /** Submit an HITL decision. POST /tasks/{id}/hitl/decision */
  hitlDecision: (
    id: string,
    data: HITLDecisionRequest
  ): Promise<HITLDecisionResponse> =>
    request<HITLDecisionResponse>(`/tasks/${id}/hitl/decision`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

export { ApiError };
