/**
 * Custom WebSocket hook for live trace event streaming.
 *
 * Connects to ws://localhost:3100/events/stream?task_id={id}
 * Auto-reconnects on drop with exponential backoff (1s, 2s, 4s, max 30s).
 * Backfills missed events on reconnect by requesting GET /tasks/{id}
 * and populating the trace_history from the response.
 * Stops reconnection when a task_complete event is received.
 * Returns { events, status, error }.
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { TraceEvent, TaskDetail } from "@/types/api";

type ConnectionStatus = "connecting" | "connected" | "disconnected";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3100";

const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

const MAX_BACKOFF_MS = 30_000;
const INITIAL_BACKOFF_MS = 1_000;

interface UseTraceStreamReturn {
  events: TraceEvent[];
  status: ConnectionStatus;
  error: string | null;
}

export function useTraceStream(taskId: string | null): UseTraceStreamReturn {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const taskIdRef = useRef(taskId);
  const mountedRef = useRef(true);
  const isTerminalRef = useRef(false);
  const seenSpanIdsRef = useRef<Set<string>>(new Set());

  // Keep taskId ref current
  useEffect(() => {
    taskIdRef.current = taskId;
  }, [taskId]);

  // Reset terminal state when taskId changes
  useEffect(() => {
    isTerminalRef.current = false;
    seenSpanIdsRef.current = new Set();
  }, [taskId]);

  const backfill = useCallback(async (id: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${id}`);
      if (response.ok) {
        const taskData = (await response.json()) as TaskDetail;

        // If the task is in a terminal state, mark as terminal
        const terminalStates = ["completed", "failed", "rejected"];
        if (taskData.status && terminalStates.includes(taskData.status)) {
          isTerminalRef.current = true;
        }

        // Backfill from trace_history if available
        if (taskData.trace_history && Array.isArray(taskData.trace_history)) {
          const backfillEvents = taskData.trace_history as TraceEvent[];
          setEvents((prev) => {
            // Merge backfilled events, avoiding duplicates
            const existingIds = new Set(prev.map((e) => e.span_id));
            const newEvents = backfillEvents.filter(
              (e) => !existingIds.has(e.span_id)
            );
            if (newEvents.length === 0) return prev;

            // Combine and sort by started_at timestamp
            const combined = [...prev, ...newEvents];
            combined.sort((a, b) => {
              const aTime = a.started_at ? new Date(a.started_at).getTime() : 0;
              const bTime = b.started_at ? new Date(b.started_at).getTime() : 0;
              return aTime - bTime;
            });

            // Update seen span IDs
            for (const e of combined) {
              seenSpanIdsRef.current.add(e.span_id);
            }

            return combined;
          });

          // Check for task_complete in backfilled events
          const hasComplete = backfillEvents.some(
            (e) => e.type === "task_complete"
          );
          if (hasComplete) {
            isTerminalRef.current = true;
          }
        }
      }
    } catch {
      // Silently ignore backfill errors — live stream will resume
    }
  }, []);

  const connect = useCallback(() => {
    if (!taskIdRef.current || !mountedRef.current) return;

    // Don't reconnect if task is in terminal state
    if (isTerminalRef.current) return;

    const id = taskIdRef.current;

    const wsUrl = `${WS_BASE_URL}/events/stream?task_id=${id}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    setStatus("connecting");

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setStatus("connected");
      setError(null);
      backoffRef.current = INITIAL_BACKOFF_MS;
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(event.data as string) as TraceEvent;

        // Skip ping messages
        if (data.type === "ping") return;

        // Skip duplicate events (same span_id already seen)
        if (data.span_id && seenSpanIdsRef.current.has(data.span_id)) {
          return;
        }

        // Track the span_id
        if (data.span_id) {
          seenSpanIdsRef.current.add(data.span_id);
        }

        // Handle task_complete — mark as final and stop reconnection
        if (data.type === "task_complete") {
          isTerminalRef.current = true;
          setEvents((prev) => [...prev, data]);
          return;
        }

        setEvents((prev) => [...prev, data]);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setError("WebSocket connection error");
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus("disconnected");
      wsRef.current = null;

      // Don't reconnect if task is in terminal state
      if (isTerminalRef.current) return;

      // Schedule reconnect with exponential backoff
      const delay = backoffRef.current;
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);

      reconnectTimeoutRef.current = setTimeout(() => {
        if (mountedRef.current && taskIdRef.current) {
          void backfill(taskIdRef.current);
          connect();
        }
      }, delay);
    };
  }, [backfill]);

  useEffect(() => {
    mountedRef.current = true;

    if (taskId) {
      // Backfill on initial connection
      void backfill(taskId);
      connect();
    }

    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on intentional close
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [taskId, connect, backfill]);

  // Reset events when taskId changes
  useEffect(() => {
    setEvents([]);
    setError(null);
  }, [taskId]);

  return { events, status, error };
}
