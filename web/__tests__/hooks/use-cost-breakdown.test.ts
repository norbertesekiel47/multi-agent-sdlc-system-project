/**
 * Tests for the useCostBreakdown hook.
 *
 * Verifies per-agent cost aggregation from trace events,
 * including: per-agent breakdown, totals, live updates,
 * frozen state for terminal tasks, WS reconnect re-sync.
 */

import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useCostBreakdown } from "@/hooks/use-cost-breakdown";
import type { TraceEvent } from "@/types/api";

// Helper to create a node_end trace event with cost data
function makeNodeEndEvent(
  agent: string,
  tokensIn: number,
  tokensOut: number,
  cachedTokens: number,
  costUsd: number,
  spanId = ""
): TraceEvent {
  return {
    type: "node_end",
    task_id: "task-1",
    span_id: spanId || `span-${agent}-${Math.random()}`,
    parent_span_id: null,
    name: agent,
    started_at: "2026-05-24T10:00:00Z",
    ended_at: "2026-05-24T10:00:05Z",
    tokens_in: tokensIn,
    tokens_out: tokensOut,
    cost_usd: costUsd,
    status: "ok",
    agent,
    cached_tokens: cachedTokens,
  };
}

function makeNodeStartEvent(agent: string, spanId = ""): TraceEvent {
  return {
    type: "node_start",
    task_id: "task-1",
    span_id: spanId || `span-start-${agent}-${Math.random()}`,
    parent_span_id: null,
    name: agent,
    started_at: "2026-05-24T10:00:00Z",
    ended_at: null,
    tokens_in: null,
    tokens_out: null,
    cost_usd: null,
    status: null,
    agent,
    cached_tokens: null,
  };
}

function makeTaskCompleteEvent(): TraceEvent {
  return {
    type: "task_complete",
    task_id: "task-1",
    span_id: "span-complete",
    parent_span_id: null,
    name: "task_complete",
    started_at: "2026-05-24T10:05:00Z",
    ended_at: "2026-05-24T10:05:00Z",
    tokens_in: null,
    tokens_out: null,
    cost_usd: null,
    status: "ok",
  };
}

describe("useCostBreakdown", () => {
  it("returns zero totals for empty events", () => {
    const { result } = renderHook(() => useCostBreakdown([], false));
    const { totals, perAgent } = result.current;

    expect(totals.tokensIn).toBe(0);
    expect(totals.tokensOut).toBe(0);
    expect(totals.cachedTokens).toBe(0);
    expect(totals.costUsd).toBeCloseTo(0);
    expect(Object.keys(perAgent).length).toBe(0);
  });

  it("aggregates per-agent costs from node_end events", () => {
    const events: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
      makeNodeEndEvent("coder", 2000, 500, 300, 0.15),
      makeNodeEndEvent("reviewer", 1500, 300, 200, 0.10),
    ];

    const { result } = renderHook(() => useCostBreakdown(events, false));
    const { totals, perAgent } = result.current;

    // Per-agent breakdown
    expect(perAgent.planner).toBeDefined();
    expect(perAgent.planner.tokensIn).toBe(1000);
    expect(perAgent.planner.tokensOut).toBe(200);
    expect(perAgent.planner.cachedTokens).toBe(0);
    expect(perAgent.planner.costUsd).toBeCloseTo(0.05);

    expect(perAgent.coder).toBeDefined();
    expect(perAgent.coder.tokensIn).toBe(2000);
    expect(perAgent.coder.tokensOut).toBe(500);
    expect(perAgent.coder.cachedTokens).toBe(300);
    expect(perAgent.coder.costUsd).toBeCloseTo(0.15);

    expect(perAgent.reviewer).toBeDefined();
    expect(perAgent.reviewer.tokensIn).toBe(1500);
    expect(perAgent.reviewer.tokensOut).toBe(300);
    expect(perAgent.reviewer.cachedTokens).toBe(200);
    expect(perAgent.reviewer.costUsd).toBeCloseTo(0.10);

    // Totals
    expect(totals.tokensIn).toBe(4500);
    expect(totals.tokensOut).toBe(1000);
    expect(totals.cachedTokens).toBe(500);
    expect(totals.costUsd).toBeCloseTo(0.30);
  });

  it("sums multiple turns from same agent", () => {
    const events: TraceEvent[] = [
      makeNodeEndEvent("coder", 2000, 500, 300, 0.15, "span-coder-1"),
      makeNodeEndEvent("coder", 1800, 400, 250, 0.12, "span-coder-2"),
    ];

    const { result } = renderHook(() => useCostBreakdown(events, false));
    const { perAgent, totals } = result.current;

    expect(perAgent.coder.tokensIn).toBe(3800);
    expect(perAgent.coder.tokensOut).toBe(900);
    expect(perAgent.coder.cachedTokens).toBe(550);
    expect(perAgent.coder.costUsd).toBeCloseTo(0.27);

    expect(totals.tokensIn).toBe(3800);
    expect(totals.costUsd).toBeCloseTo(0.27);
  });

  it("ignores node_start and other non-cost events", () => {
    const events: TraceEvent[] = [
      makeNodeStartEvent("planner"),
      makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
      makeNodeStartEvent("coder"),
    ];

    const { result } = renderHook(() => useCostBreakdown(events, false));
    const { perAgent } = result.current;

    // Only planner has a node_end, coder only has node_start (no cost data yet)
    expect(perAgent.planner).toBeDefined();
    expect(perAgent.coder).toBeUndefined();
  });

  it("handles null cost fields gracefully", () => {
    const event: TraceEvent = {
      type: "node_end",
      task_id: "task-1",
      span_id: "span-1",
      parent_span_id: null,
      name: "planner",
      started_at: "2026-05-24T10:00:00Z",
      ended_at: "2026-05-24T10:00:05Z",
      tokens_in: null,
      tokens_out: null,
      cost_usd: null,
      status: "ok",
      agent: "planner",
      cached_tokens: null,
    };

    const { result } = renderHook(() => useCostBreakdown([event], false));
    const { perAgent } = result.current;

    expect(perAgent.planner).toBeDefined();
    expect(perAgent.planner.tokensIn).toBe(0);
    expect(perAgent.planner.tokensOut).toBe(0);
    expect(perAgent.planner.cachedTokens).toBe(0);
    expect(perAgent.planner.costUsd).toBeCloseTo(0);
  });

  it("detects budget threshold: 80% warning", () => {
    // MAX_COST_PER_TASK_USD = 2.00, 80% = 1.60
    const events: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 1.60),
    ];

    const { result } = renderHook(() =>
      useCostBreakdown(events, false, 2.0)
    );

    expect(result.current.isBudgetWarning).toBe(true);
    expect(result.current.isBudgetExceeded).toBe(false);
  });

  it("detects budget threshold: 100% exceeded", () => {
    const events: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 2.01),
    ];

    const { result } = renderHook(() =>
      useCostBreakdown(events, false, 2.0)
    );

    expect(result.current.isBudgetWarning).toBe(true);
    expect(result.current.isBudgetExceeded).toBe(true);
  });

  it("does not flag budget warning when cost is below 80%", () => {
    const events: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 1.50),
    ];

    const { result } = renderHook(() =>
      useCostBreakdown(events, false, 2.0)
    );

    expect(result.current.isBudgetWarning).toBe(false);
    expect(result.current.isBudgetExceeded).toBe(false);
  });

  it("freezes cost when task is terminal", () => {
    const events: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
      makeTaskCompleteEvent(),
    ];

    const { result } = renderHook(() => useCostBreakdown(events, true));
    const frozen = result.current.totals;

    // Even if more events were somehow added, the hook should
    // stop updating when isTerminal is true
    expect(frozen.tokensIn).toBe(1000);
    expect(frozen.costUsd).toBeCloseTo(0.05);
  });

  it("per-agent breakdown sums to total within rounding", () => {
    const events: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
      makeNodeEndEvent("coder", 2000, 500, 300, 0.15),
      makeNodeEndEvent("reviewer", 1500, 300, 200, 0.10),
      makeNodeEndEvent("qa", 800, 100, 50, 0.03),
    ];

    const { result } = renderHook(() => useCostBreakdown(events, false));
    const { totals, perAgent } = result.current;

    // Sum of per-agent USD should equal total within $0.01
    const agentCostSum = Object.values(perAgent).reduce(
      (sum, a) => sum + a.costUsd,
      0
    );
    expect(Math.abs(agentCostSum - totals.costUsd)).toBeLessThan(0.01);

    // Sum of per-agent tokens should equal total
    const agentTokensInSum = Object.values(perAgent).reduce(
      (sum, a) => sum + a.tokensIn,
      0
    );
    expect(agentTokensInSum).toBe(totals.tokensIn);

    const agentTokensOutSum = Object.values(perAgent).reduce(
      (sum, a) => sum + a.tokensOut,
      0
    );
    expect(agentTokensOutSum).toBe(totals.tokensOut);

    const agentCachedSum = Object.values(perAgent).reduce(
      (sum, a) => sum + a.cachedTokens,
      0
    );
    expect(agentCachedSum).toBe(totals.cachedTokens);
  });

  it("cost values are strictly non-decreasing across updates", () => {
    const { result, rerender } = renderHook(
      ({ events, isTerminal }) => useCostBreakdown(events, isTerminal),
      { initialProps: { events: [] as TraceEvent[], isTerminal: false } }
    );

    // Initial: zero
    expect(result.current.totals.costUsd).toBeCloseTo(0);

    // First event
    const events1 = [makeNodeEndEvent("planner", 1000, 200, 0, 0.05)];
    rerender({ events: events1, isTerminal: false });
    const cost1 = result.current.totals.costUsd;
    expect(cost1).toBeCloseTo(0.05);

    // Second event
    const events2 = [
      ...events1,
      makeNodeEndEvent("coder", 2000, 500, 300, 0.15),
    ];
    rerender({ events: events2, isTerminal: false });
    const cost2 = result.current.totals.costUsd;
    expect(cost2).toBeGreaterThanOrEqual(cost1);
    expect(cost2).toBeCloseTo(0.20);
  });

  it("WS reconnect re-syncs cost without regression", () => {
    // Simulate initial events
    const preDropEvents: TraceEvent[] = [
      makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
      makeNodeEndEvent("coder", 2000, 500, 300, 0.15),
    ];

    const { result, rerender } = renderHook(
      ({ events, isTerminal }) => useCostBreakdown(events, isTerminal),
      { initialProps: { events: preDropEvents, isTerminal: false } }
    );

    const preDropCost = result.current.totals.costUsd;
    expect(preDropCost).toBeCloseTo(0.20);

    // Simulate reconnect: backfilled events (same as before + new)
    const postReconnectEvents: TraceEvent[] = [
      // Backfilled (same span_ids as before — duplicates should not double-count)
      makeNodeEndEvent("planner", 1000, 200, 0, 0.05),
      makeNodeEndEvent("coder", 2000, 500, 300, 0.15),
      // New event after reconnect
      makeNodeEndEvent("reviewer", 1500, 300, 200, 0.10),
    ];

    rerender({ events: postReconnectEvents, isTerminal: false });

    // Cost should include all events, no regression
    expect(result.current.totals.costUsd).toBeGreaterThanOrEqual(preDropCost);
    expect(result.current.totals.costUsd).toBeCloseTo(0.30);
    expect(result.current.totals.tokensIn).toBe(4500);
  });
});
