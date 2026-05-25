"use client";

import { useState } from "react";
import Link from "next/link";
import { useTasks } from "@/hooks/use-tasks";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import type { TaskStatus } from "@/types/api";

const OUTCOME_OPTIONS = [
  { value: "all", label: "All Outcomes" },
  { value: "pr_opened", label: "PR Opened" },
  { value: "hitl_rejected", label: "HITL Rejected" },
  { value: "loop_detected", label: "Loop Detected" },
  { value: "retry_budget_exhausted", label: "Retry Budget Exhausted" },
  { value: "cost_budget_exhausted", label: "Cost Budget Exhausted" },
  { value: "guardrail_block", label: "Guardrail Block" },
  { value: "uncertainty_escalation", label: "Uncertainty Escalation" },
];

const TOPOLOGY_OPTIONS = [
  { value: "all", label: "All Topologies" },
  { value: "single_agent", label: "Single Agent" },
  { value: "supervisor_only", label: "Supervisor Only" },
  { value: "hybrid", label: "Hybrid" },
];

function formatCost(cost: number | null): string {
  if (cost === null || cost === undefined) return "—";
  return `$${cost.toFixed(4)}`;
}

function formatDuration(
  startedAt: string | null,
  endedAt: string | null
): string {
  if (!startedAt || !endedAt) return "—";
  const ms =
    new Date(endedAt).getTime() - new Date(startedAt).getTime();
  if (ms < 0) return "—";
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return date.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const statusColors: Record<TaskStatus, string> = {
  running: "bg-blue-500/15 text-blue-500",
  awaiting_hitl: "bg-yellow-500/15 text-yellow-500",
  approved: "bg-green-500/15 text-green-500",
  rejected: "bg-red-500/15 text-red-500",
  completed: "bg-green-500/15 text-green-500",
  failed: "bg-red-500/15 text-red-500",
};

export function HistoryList() {
  const [outcomeFilter, setOutcomeFilter] = useState<string>("all");
  const [topologyFilter, setTopologyFilter] = useState<string>("all");
  const [repoFilter, setRepoFilter] = useState("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 20;

  // Build query params for terminal tasks
  const params: Record<string, string | number | undefined> = {
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  if (outcomeFilter !== "all") params.outcome = outcomeFilter;
  if (topologyFilter !== "all") params.topology = topologyFilter;
  if (repoFilter.trim()) params.repo_url = repoFilter.trim();

  const { data, isLoading, error } = useTasks({
    ...params,
    status: "completed,failed,rejected",
    refetchInterval: 10_000,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Task History</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Filters */}
        <div className="flex flex-wrap gap-3">
          <Select
            value={outcomeFilter}
            onValueChange={(v) => {
              setOutcomeFilter(v);
              setPage(0);
            }}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {OUTCOME_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={topologyFilter}
            onValueChange={(v) => {
              setTopologyFilter(v);
              setPage(0);
            }}
          >
            <SelectTrigger className="w-[160px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TOPOLOGY_OPTIONS.map((t) => (
                <SelectItem key={t.value} value={t.value}>
                  {t.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Input
            placeholder="Filter by repo…"
            value={repoFilter}
            onChange={(e) => {
              setRepoFilter(e.target.value);
              setPage(0);
            }}
            className="w-56"
          />
        </div>

        {/* Error state */}
        {error && (
          <p className="text-sm text-muted-foreground">
            Unable to load task history. Please try again.
          </p>
        )}

        {/* Loading state */}
        {isLoading && (
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        )}

        {/* Empty state */}
        {!isLoading && !error && (!data?.tasks.length) && (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No tasks match these filters
          </p>
        )}

        {/* Results table */}
        {!isLoading && data?.tasks.length ? (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Repo</TableHead>
                  <TableHead>Issue</TableHead>
                  <TableHead>Topology</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Cost</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Ended</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.tasks.map((task) => (
                  <TableRow key={task.id}>
                    <TableCell className="font-medium">
                      <Link
                        href={`/tasks/${task.id}`}
                        className="text-primary hover:underline"
                      >
                        {task.repo_url.replace("https://github.com/", "")}
                      </Link>
                    </TableCell>
                    <TableCell>#{task.issue_number ?? "—"}</TableCell>
                    <TableCell className="text-xs">
                      {task.topology}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={statusColors[task.status] ?? ""}
                      >
                        {task.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{formatCost(task.total_cost_usd)}</TableCell>
                    <TableCell>
                      {formatDuration(task.started_at, task.ended_at)}
                    </TableCell>
                    <TableCell>{formatDate(task.ended_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            {/* Pagination */}
            <div className="flex items-center justify-between pt-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </Button>
              <span className="text-xs text-muted-foreground">
                Page {page + 1}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={(data?.tasks.length ?? 0) < PAGE_SIZE}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
