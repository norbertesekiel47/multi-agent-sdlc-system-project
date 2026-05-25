/**
 * Per-agent Cost Panel for /tasks/[id].
 *
 * Shows:
 *   - Running totals for tokens-in, tokens-out, cached-tokens, USD cost
 *   - Per-agent breakdown (Planner, Coder, Reviewer, QA, Supervisor)
 *     where each row has tokens-in, tokens-out, cached-tokens, cost
 *   - 80% budget warning banner
 *   - 100% budget halt banner
 *   - Frozen indicator for terminal tasks
 *
 * Data is derived from WebSocket trace events via useCostBreakdown,
 * which aggregates per-agent costs from node_end events.
 * On reconnect, events are backfilled so costs re-sync without regression.
 *
 * Formatting:
 *   - USD: $X.XXXX with optional comma thousands (e.g. $1,234.5678)
 *   - Tokens: comma thousands (e.g. 1,000,000)
 *   - Never shows NaN, undefined, or scientific notation
 */

"use client";

import { useMemo } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useCostBreakdown } from "@/hooks/use-cost-breakdown";
import type { TraceEvent, TaskDetail, AgentCostRow } from "@/types/api";

interface CostPanelProps {
  task: TaskDetail | undefined;
  /** Trace events from the WebSocket stream (for live per-agent breakdown) */
  events?: TraceEvent[];
  /** Whether the task is in a terminal state (completed, failed, rejected) */
  isTerminal?: boolean;
  /** Maximum budget in USD (default $2.00) */
  maxBudgetUsd?: number;
}

const DEFAULT_MAX_BUDGET = 2.0;

/** Agent display order — consistent across topologies */
const AGENT_ORDER = ["planner", "coder", "reviewer", "qa", "supervisor", "single_agent"];

/** Format USD as $X.XXXX with optional comma thousands */
function formatCost(cost: number | string | null | undefined): string {
  if (cost === null || cost === undefined) return "—";
  const numCost = typeof cost === "string" ? parseFloat(cost) : cost;
  if (isNaN(numCost)) return "—";
  const formatted = numCost.toLocaleString("en-US", {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  });
  return `$${formatted}`;
}

/** Format token counts with comma thousands */
function formatTokens(n: number | string | null | undefined): string {
  if (n === null || n === undefined) return "—";
  const numN = typeof n === "string" ? parseInt(n, 10) : n;
  if (isNaN(numN)) return "—";
  return numN.toLocaleString("en-US");
}

/** Capitalize agent name for display */
function displayAgent(agent: string): string {
  return agent.charAt(0).toUpperCase() + agent.slice(1);
}

/** Merged per-agent row from both trace events and DB agent_costs */
interface MergedAgentRow {
  tokensIn: number;
  tokensOut: number;
  cachedTokens: number;
  costUsd: number;
}

/** Merge trace-event-derived per-agent data with DB agent_costs.
 *  Trace events take priority (more granular, fresher), DB fills gaps
 *  (available for terminal tasks where no events are flowing).
 */
function mergeAgentData(
  tracePerAgent: Record<string, { tokensIn: number; tokensOut: number; cachedTokens: number; costUsd: number }>,
  dbAgentCosts: Record<string, AgentCostRow> | null | undefined,
): Record<string, MergedAgentRow> {
  const merged: Record<string, MergedAgentRow> = {};

  // Start with trace-derived data
  for (const [agent, row] of Object.entries(tracePerAgent)) {
    merged[agent] = {
      tokensIn: row.tokensIn,
      tokensOut: row.tokensOut,
      cachedTokens: row.cachedTokens,
      costUsd: row.costUsd,
    };
  }

  // Merge DB agent_costs (fills gaps for terminal tasks)
  if (dbAgentCosts) {
    for (const [agent, row] of Object.entries(dbAgentCosts)) {
      const existing = merged[agent];
      const dbCostUsd = parseFloat(String(row.cost_usd ?? "0"));
      const dbTokensIn = Number(row.tokens_in ?? 0);
      const dbTokensOut = Number(row.tokens_out ?? 0);
      const dbCached = Number(row.cached_tokens ?? 0);

      if (existing) {
        // Use the larger values (DB values should equal trace-derived for consistency)
        merged[agent] = {
          tokensIn: Math.max(existing.tokensIn, dbTokensIn),
          tokensOut: Math.max(existing.tokensOut, dbTokensOut),
          cachedTokens: Math.max(existing.cachedTokens, dbCached),
          costUsd: Math.max(existing.costUsd, dbCostUsd),
        };
      } else {
        merged[agent] = {
          tokensIn: dbTokensIn,
          tokensOut: dbTokensOut,
          cachedTokens: dbCached,
          costUsd: dbCostUsd,
        };
      }
    }
  }

  return merged;
}

export function CostPanel({
  task,
  events = [],
  isTerminal = false,
  maxBudgetUsd = DEFAULT_MAX_BUDGET,
}: CostPanelProps) {
  // Compute per-agent breakdown from trace events
  const { totals, perAgent, isBudgetWarning, isBudgetExceeded } =
    useCostBreakdown(events, isTerminal, maxBudgetUsd);

  // Merge trace-derived per-agent data with DB agent_costs
  const mergedAgents = useMemo(
    () => mergeAgentData(perAgent, task?.agent_costs ?? null),
    [perAgent, task?.agent_costs],
  );

  // Sort agents by display order
  const sortedAgents = useMemo(() => {
    const agents = Object.keys(mergedAgents);
    return agents.sort((a, b) => {
      const ai = AGENT_ORDER.indexOf(a);
      const bi = AGENT_ORDER.indexOf(b);
      return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
    });
  }, [mergedAgents]);

  // Loading state
  if (!task) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Cost</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Loading cost data…
          </p>
        </CardContent>
      </Card>
    );
  }

  const hasAgentData = sortedAgents.length > 0;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base">Cost</CardTitle>
        {isTerminal && (
          <Badge variant="outline" className="text-xs text-muted-foreground">
            Final
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Budget warning banners */}
        {isBudgetExceeded && (
          <div className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            ⛔ Budget exceeded — task halted at 100% of ${maxBudgetUsd.toFixed(2)} budget
          </div>
        )}
        {isBudgetWarning && !isBudgetExceeded && (
          <div className="rounded-md border border-yellow-500/50 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-400">
            ⚠️ 80% budget warning — ${totals.costUsd.toFixed(4)} of ${maxBudgetUsd.toFixed(2)} used
          </div>
        )}

        {/* Running totals summary — prefer task-level data (from DB), fall back to trace-derived totals */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <span className="text-muted-foreground">Total Cost</span>
          <span className="font-mono text-right" data-testid="total-cost">
            {task.total_cost_usd !== null && task.total_cost_usd !== undefined
              ? formatCost(task.total_cost_usd)
              : totals.costUsd > 0
                ? formatCost(totals.costUsd)
                : "—"}
          </span>

          <span className="text-muted-foreground">Tokens In</span>
          <span className="font-mono text-right" data-testid="total-tokens-in">
            {task.total_tokens_in !== null && task.total_tokens_in !== undefined
              ? formatTokens(task.total_tokens_in)
              : totals.tokensIn > 0
                ? formatTokens(totals.tokensIn)
                : "—"}
          </span>

          <span className="text-muted-foreground">Tokens Out</span>
          <span className="font-mono text-right" data-testid="total-tokens-out">
            {task.total_tokens_out !== null && task.total_tokens_out !== undefined
              ? formatTokens(task.total_tokens_out)
              : totals.tokensOut > 0
                ? formatTokens(totals.tokensOut)
                : "—"}
          </span>

          <span className="text-muted-foreground">Cached Tokens</span>
          <span className="font-mono text-right" data-testid="total-cached-tokens">
            {task.total_tokens_cached !== null && task.total_tokens_cached !== undefined
              ? formatTokens(task.total_tokens_cached)
              : totals.cachedTokens > 0
                ? formatTokens(totals.cachedTokens)
                : "—"}
          </span>
        </div>

        {/* Per-agent breakdown table */}
        {hasAgentData && (
          <>
            <Table aria-label="Cost breakdown">
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Agent</TableHead>
                  <TableHead className="text-right text-xs">Tokens In</TableHead>
                  <TableHead className="text-right text-xs">Tokens Out</TableHead>
                  <TableHead className="text-right text-xs">Cached</TableHead>
                  <TableHead className="text-right text-xs">Cost</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedAgents.map((agent) => {
                  const row = mergedAgents[agent];
                  return (
                    <TableRow key={agent}>
                      <TableCell className="font-medium text-sm">
                        {displayAgent(agent)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {formatTokens(row.tokensIn)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {formatTokens(row.tokensOut)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {row.cachedTokens > 0
                          ? formatTokens(row.cachedTokens)
                          : "—"}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {formatCost(row.costUsd)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </>
        )}

        {/* Empty state when no agent data yet */}
        {!hasAgentData && !isTerminal && (
          <p className="text-xs text-muted-foreground">
            Per-agent breakdown will appear as agents complete turns.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
