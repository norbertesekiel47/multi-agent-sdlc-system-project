/**
 * Tests for the useTraceStream hook.
 *
 * Verifies: initial state, connection, reconnection behavior.
 * Uses mocked WebSocket.
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

// Store the original WebSocket
const originalWebSocket = globalThis.WebSocket;

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.useFakeTimers();
  // @ts-expect-error Mock WebSocket
  globalThis.WebSocket = MockWebSocket;
});

afterEach(() => {
  globalThis.WebSocket = originalWebSocket;
  vi.useRealTimers();
});

describe("useTraceStream", () => {
  it("returns initial disconnected state when no taskId", () => {
    const { result } = renderHook(() => useTraceStream(null));

    expect(result.current.status).toBe("disconnected");
    expect(result.current.events).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it("connects to WebSocket when taskId is provided", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    const ws = MockWebSocket.instances[0];
    expect(ws.url).toContain("task_id=task-123");
    expect(ws.url).toContain("ws://");
  });

  it("updates status to connected when WebSocket opens", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    expect(result.current.status).toBe("connected");
    expect(result.current.error).toBeNull();
  });

  it("receives trace events from WebSocket", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    const traceEvent = {
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
    };

    act(() => {
      ws.simulateMessage(traceEvent);
    });

    expect(result.current.events).toHaveLength(1);
    expect(result.current.events[0].type).toBe("node_start");
    expect(result.current.events[0].name).toBe("planner");
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

  it("updates status to disconnected on WebSocket close", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    act(() => {
      ws.simulateClose();
    });

    expect(result.current.status).toBe("disconnected");
  });

  it("attempts reconnection after disconnect", () => {
    const { result } = renderHook(() => useTraceStream("task-123"));

    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.simulateOpen();
    });

    act(() => {
      ws.simulateClose();
    });

    // Advance timers for reconnect backoff
    act(() => {
      vi.advanceTimersByTime(2000);
    });

    // A new WebSocket should have been created
    expect(MockWebSocket.instances.length).toBeGreaterThan(1);
  });
});
