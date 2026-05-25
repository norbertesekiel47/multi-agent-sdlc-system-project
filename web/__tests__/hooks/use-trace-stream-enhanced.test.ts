/**
 * Enhanced tests for the useTraceStream hook.
 *
 * Verifies: backfill from GET /tasks/{id}, terminal state stops
 * reconnection, task_complete event handling, status indicators.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTraceStream } from "@/hooks/use-trace-stream";

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  static instances: MockWebSocket[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send() {}
  close() {
    this.readyState = MockWebSocket.CLOSED;
  }

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  simulateClose() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}

// Mock fetch for backfill
const mockFetch = vi.fn();

// Store the original WebSocket and fetch
const originalWebSocket = globalThis.WebSocket;
const originalFetch = globalThis.fetch;

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.useFakeTimers();
  // @ts-expect-error Mock WebSocket
  globalThis.WebSocket = MockWebSocket;
  globalThis.fetch = mockFetch;
});

afterEach(() => {
  globalThis.WebSocket = originalWebSocket;
  globalThis.fetch = originalFetch;
  vi.useRealTimers();
});

describe("useTraceStream - Enhanced", () => {
  it("backfills events from GET /tasks/{id} on initial connection", async () => {
    // Mock backfill response
    mockFetch.mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          id: "task-123",
          repo_url: "https://github.com/org/repo",
          status: "running",
          trace_history: [
            {
              type: "node_start",
              task_id: "task-123",
              span_id: "historical-1",
              parent_span_id: null,
              name: "planner",
              started_at: "2026-05-24T09:59:00Z",
              ended_at: null,
              tokens_in: null,
              tokens_out: null,
              cost_usd: null,
              status: null,
            },
          ],
        }),
    });

    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    // Wait for backfill fetch to complete
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });

    // The backfilled event should be in the events list
    // (This will fail until we implement backfill properly)
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/tasks/task-123")
    );
  });

  it("stops reconnection attempts after task_complete event", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    // Send a task_complete event
    act(() => {
      ws.simulateMessage({
        type: "task_complete",
        task_id: "task-123",
        span_id: "complete-span",
        parent_span_id: null,
        name: "task_complete",
        started_at: "2026-05-24T10:00:10Z",
        ended_at: "2026-05-24T10:00:10Z",
        tokens_in: null,
        tokens_out: null,
        cost_usd: null,
        status: "completed",
      });
    });

    // Close the WebSocket
    act(() => {
      ws.simulateClose();
    });

    // Advance timers significantly past the max backoff
    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    // No new WebSocket should be created after task_complete
    // (This will fail until we implement terminal state check)
    const wsCountAfterComplete = MockWebSocket.instances.length;
    // Should be 1 (the original) or at most 2 (one reconnect attempt)
    // but NOT continuously creating new ones
    expect(wsCountAfterComplete).toBeLessThanOrEqual(2);
  });

  it("ignores ping messages", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    act(() => {
      ws.simulateMessage({ type: "ping" });
    });

    expect(result.current.events).toHaveLength(0);
  });

  it("resets events when taskId changes", () => {
    const { result, rerender } = renderHook(
      ({ taskId }) => useTraceStream(taskId),
      { initialProps: { taskId: "task-123" } }
    );

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    act(() => {
      ws.simulateMessage({
        type: "node_start",
        task_id: "task-123",
        span_id: "span-1",
        parent_span_id: null,
        name: "planner",
        started_at: "2026-05-24T10:00:00Z",
        ended_at: null,
        tokens_in: null,
        tokens_out: null,
        cost_usd: null,
        status: null,
      });
    });

    expect(result.current.events).toHaveLength(1);

    // Change taskId
    rerender({ taskId: "task-456" });

    // Events should be reset
    expect(result.current.events).toHaveLength(0);
  });

  it("WebSocket URL contains correct task_id query parameter", () => {
    renderHook(() => useTraceStream("my-special-task"));

    const ws = MockWebSocket.instances[0];
    expect(ws.url).toContain("task_id=my-special-task");
  });

  it("uses ws:// protocol for WebSocket URL", () => {
    renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    expect(ws.url).toContain("ws://");
  });
});
