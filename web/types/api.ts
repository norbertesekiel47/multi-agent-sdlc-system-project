/**
 * TypeScript types matching the backend Pydantic schemas.
 *
 * These types mirror the FastAPI response models in src/api/models.py
 * and the episodic store schemas in src/memory/episodic/models.py.
 */

export type Topology = "single_agent" | "supervisor_only" | "hybrid";

export type TaskStatus =
  | "running"
  | "awaiting_hitl"
  | "approved"
  | "rejected"
  | "completed"
  | "failed";

export type HITLDecision = "approve" | "reject";

export interface TaskListItem {
  id: string;
  repo_url: string;
  issue_number: number | null;
  topology: Topology;
  status: TaskStatus;
  total_cost_usd: number | null;
  started_at: string;
  ended_at: string | null;
}

export interface TaskDetail {
  id: string;
  repo_url: string;
  issue_number: number | null;
  issue_text: string;
  topology: Topology;
  status: TaskStatus;
  total_cost_usd: number | null;
  total_tokens_in: number | null;
  total_tokens_out: number | null;
  total_tokens_cached: number | null;
  hitl_decision: HITLDecision | null;
  pr_url: string | null;
  started_at: string;
  ended_at: string | null;
}

export interface ListTasksResponse {
  tasks: TaskListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateTaskRequest {
  repo_url: string;
  issue_number: number;
  issue_text: string;
  topology: Topology;
  auto_start?: boolean;
}

export interface CreateTaskResponse {
  id: string;
}

export interface HITLDecisionRequest {
  decision: "approve" | "reject";
  reason?: string | null;
}

export interface HITLDecisionResponse {
  task_id: string;
  decision: "approve" | "reject";
  status: TaskStatus;
}

export interface ErrorResponse {
  error: string;
  detail?: string | null;
}

/** WebSocket trace event types */
export type TraceEventType =
  | "node_start"
  | "node_end"
  | "tool_call"
  | "llm_completion"
  | "task_complete"
  | "ping"
  | "error";

export interface TraceEvent {
  type: TraceEventType;
  task_id: string;
  span_id: string;
  parent_span_id: string | null;
  name: string;
  started_at: string | null;
  ended_at: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  cost_usd: number | null;
  status: string | null;
  agent?: string;
  cached_tokens?: number | null;
}
