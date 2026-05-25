/**
 * React Query hooks for task data.
 *
 * - useTasks: query hook for task list with refetch interval
 * - useTask: query hook for single task
 * - useHitlDecision: mutation hook for POST /tasks/{id}/hitl/decision
 */

"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, ApiError } from "@/lib/api-client";
import type {
  CreateTaskRequest,
  CreateTaskResponse,
  HITLDecisionRequest,
  HITLDecisionResponse,
  ListTasksResponse,
  TaskDetail,
} from "@/types/api";

/** Query key factory for consistent key generation */
export const taskKeys = {
  all: ["tasks"] as const,
  lists: () => [...taskKeys.all, "list"] as const,
  list: (params?: Record<string, string | number | undefined>) =>
    [...taskKeys.lists(), params] as const,
  details: () => [...taskKeys.all, "detail"] as const,
  detail: (id: string) => [...taskKeys.details(), id] as const,
};

/** Hook for listing tasks with optional filters and polling */
export function useTasks(params?: {
  repo_url?: string;
  repo?: string;
  status?: string;
  outcome?: string;
  topology?: string;
  limit?: number;
  offset?: number;
  refetchInterval?: number;
}) {
  return useQuery<ListTasksResponse, ApiError>({
    queryKey: taskKeys.list(params),
    queryFn: () => apiClient.listTasks(params),
    refetchInterval: params?.refetchInterval ?? 5_000,
  });
}

/** Hook for fetching a single task by ID */
export function useTask(id: string, options?: { refetchInterval?: number }) {
  return useQuery<TaskDetail, ApiError>({
    queryKey: taskKeys.detail(id),
    queryFn: () => apiClient.getTask(id),
    enabled: !!id,
    refetchInterval: options?.refetchInterval,
  });
}

/** Mutation hook for creating a new task */
export function useCreateTask() {
  const queryClient = useQueryClient();

  return useMutation<CreateTaskResponse, ApiError, CreateTaskRequest>({
    mutationFn: (data: CreateTaskRequest) => apiClient.createTask(data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: taskKeys.lists() });
    },
  });
}

/** Mutation hook for submitting an HITL decision */
export function useHitlDecision() {
  const queryClient = useQueryClient();

  return useMutation<
    HITLDecisionResponse,
    ApiError,
    { taskId: string; data: HITLDecisionRequest }
  >({
    mutationFn: ({ taskId, data }) => apiClient.hitlDecision(taskId, data),
    onSuccess: (_result, { taskId }) => {
      void queryClient.invalidateQueries({ queryKey: taskKeys.detail(taskId) });
      void queryClient.invalidateQueries({ queryKey: taskKeys.lists() });
    },
  });
}
