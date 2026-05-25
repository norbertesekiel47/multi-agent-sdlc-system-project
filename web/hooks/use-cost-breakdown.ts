/**
 * Custom hook that aggregates per-agent cost breakdown from trace events.
 *
 * Reads node_end events (which carry tokens_in, tokens_out, cached_tokens,
 * cost_usd, and agent metadata) and computes:
 *   - Per-agent totals (Planner, Coder, Reviewer, QA, Supervisor)
 *   - Task-level totals (sum of all agents)
 *   - Budget threshold flags (80% warning, 100% exceeded)
 *
 * The hook is the single source of truth for cost data on the dashboard.
 * On initial load and after WS reconnect, the events array is backfilled
 * from GET /tasks/{id} trace_history, so cost re-syncs naturally.
 *
 * When isTerminal is true (task in completed/failed/rejected state),
 * the computed values are frozen and will not change.
 */

"use client";

import { useMemo } from "react";
import type { TraceEvent } from "@/types/api";

export interface AgentCostRow {
  tokensIn: number;
  tokensOut: number;
  cachedTokens: number;
  costUsd: number;
}

export interface CostTotals {
  tokensIn: number;
  tokensOut: number;
  cachedTokens: number;
  costUsd: number;
}

export interface UseCostBreakdownReturn {
  totals: CostTotals;
  perAgent: Record<string, AgentCostRow>;
  isBudgetWarning: boolean;
  isBudgetExceeded: boolean;
  maxBudgetUsd: number;
}

const DEFAULT_MAX_BUDGET_USD = 2.0;

export function useCostBreakdown(
  events: TraceEvent[],
  isTerminal: boolean,
  maxBudgetUsd: number = DEFAULT_MAX_BUDGET_USD
): UseCostBreakdownReturn {
  const breakdown = useMemo(() => {
    const perAgent: Record<string, AgentCostRow> = {};

    // Only aggregate from node_end events that have an agent field
    const nodeEndEvents = events.filter(
      (e): e is TraceEvent & { agent: string } =>
        e.type === "node_end" &&
        e.agent !== undefined &&
        e.agent !== null &&
        e.agent !== ""
    );

    for (const event of nodeEndEvents) {
      const agent: string = event.agent;
      const existing = perAgent[agent] ?? {
        tokensIn: 0,
        tokensOut: 0,
        cachedTokens: 0,
        costUsd: 0,
      };

      perAgent[agent] = {
        tokensIn: existing.tokensIn + (event.tokens_in ?? 0),
        tokensOut: existing.tokensOut + (event.tokens_out ?? 0),
        cachedTokens:
          existing.cachedTokens + (event.cached_tokens ?? 0),
        costUsd:
          existing.costUsd + (event.cost_usd ?? 0),
      };
    }

    // Compute totals from per-agent sums
    const totals: CostTotals = Object.values(perAgent).reduce(
      (acc, row) => ({
        tokensIn: acc.tokensIn + row.tokensIn,
        tokensOut: acc.tokensOut + row.tokensOut,
        cachedTokens: acc.cachedTokens + row.cachedTokens,
        costUsd: acc.costUsd + row.costUsd,
      }),
      { tokensIn: 0, tokensOut: 0, cachedTokens: 0, costUsd: 0 }
    );

    const isBudgetWarning = totals.costUsd >= maxBudgetUsd * 0.8;
    const isBudgetExceeded = totals.costUsd >= maxBudgetUsd;

    return { totals, perAgent, isBudgetWarning, isBudgetExceeded };
  }, [events, isTerminal, maxBudgetUsd]);

  return {
    ...breakdown,
    maxBudgetUsd,
  };
}
