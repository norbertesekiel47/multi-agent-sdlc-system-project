"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { TaskDetail } from "@/types/api";

interface CostPanelProps {
  task: TaskDetail | undefined;
}

function formatCost(cost: number | null | undefined): string {
  if (cost === null || cost === undefined) return "—";
  return `$${cost.toFixed(4)}`;
}

function formatTokens(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export function CostPanel({ task }: CostPanelProps) {
  if (!task) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Cost</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Loading cost data…</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Cost</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <span className="text-muted-foreground">Total Cost</span>
          <span className="font-mono">{formatCost(task.total_cost_usd)}</span>

          <span className="text-muted-foreground">Tokens In</span>
          <span className="font-mono">
            {formatTokens(task.total_tokens_in)}
          </span>

          <span className="text-muted-foreground">Tokens Out</span>
          <span className="font-mono">
            {formatTokens(task.total_tokens_out)}
          </span>

          <span className="text-muted-foreground">Cached Tokens</span>
          <span className="font-mono">
            {formatTokens(task.total_tokens_cached)}
          </span>
        </div>

        <Separator />

        <div className="text-xs text-muted-foreground">
          Per-agent breakdown will appear as agents complete turns.
        </div>
      </CardContent>
    </Card>
  );
}
