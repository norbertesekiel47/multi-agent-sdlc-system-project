"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
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
import type { TaskListItem, TaskStatus } from "@/types/api";

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

const TERMINAL_STATUSES: readonly string[] = ["completed", "failed", "rejected"];

const outcomeColors: Record<string, string> = {
  pr_opened: "bg-green-500/15 text-green-500",
  success: "bg-green-500/15 text-green-500",
  hitl_rejected: "bg-red-500/15 text-red-500",
  loop_detected: "bg-orange-500/15 text-orange-500",
  retry_budget_exhausted: "bg-orange-500/15 text-orange-500",
  cost_budget_exhausted: "bg-orange-500/15 text-orange-500",
  guardrail_block: "bg-red-500/15 text-red-500",
  uncertainty_escalation: "bg-orange-500/15 text-orange-500",
  sandbox_failure: "bg-red-500/15 text-red-500",
};

const statusColors: Record<TaskStatus, string> = {
  running: "bg-blue-500/15 text-blue-500",
  awaiting_hitl: "bg-yellow-500/15 text-yellow-500",
  approved: "bg-green-500/15 text-green-500",
  rejected: "bg-red-500/15 text-red-500",
  completed: "bg-green-500/15 text-green-500",
  failed: "bg-red-500/15 text-red-500",
};

function formatCost(cost: number | string | null): string {
  if (cost === null || cost === undefined) return "—";
  const numCost = typeof cost === "string" ? parseFloat(cost) : cost;
  if (isNaN(numCost)) return "—";
  return `$${numCost.toFixed(4)}`;
}

function formatDuration(
  startedAt: string | null,
  endedAt: string | null
): string {
  if (!startedAt || !endedAt) return "—";
  const ms = new Date(endedAt).getTime() - new Date(startedAt).getTime();
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

/** Shorten a UUID for display in the table */
function shortId(id: string): string {
  return id.length > 8 ? id.slice(0, 8) + "…" : id;
}

export function HistoryList() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Read initial filter values from URL search params
  const [page, setPage] = useState(() => {
    const p = searchParams.get("page");
    return p ? Math.max(0, parseInt(p, 10) || 0) : 0;
  });
  const PAGE_SIZE = 20;

  // Derive filters from URL params for AND semantics
  const outcomeFilter = searchParams.get("outcome") ?? "all";
  const topologyFilter = searchParams.get("topology") ?? "all";
  const repoFilter = searchParams.get("repo") ?? "";

  /** Update URL search params to preserve all active filters */
  const updateParams = useCallback(
    (updates: Record<string, string | null>) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(updates)) {
        if (value === null || value === "" || value === "all") {
          next.delete(key);
        } else {
          next.set(key, value);
        }
      }
      const qs = next.toString();
      router.replace(`/history${qs ? `?${qs}` : ""}`, { scroll: false });
    },
    [router, searchParams]
  );

  // Sync page with URL when it changes
  useEffect(() => {
    const currentPage = searchParams.get("page");
    const urlPage = currentPage ? Math.max(0, parseInt(currentPage, 10) || 0) : 0;
    if (urlPage !== page) {
      setPage(urlPage);
    }
  }, [searchParams, page]);

  // Build query params for terminal tasks — AND semantics
  const params: Record<string, string | number | undefined> = {
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  // Only pass status=terminal to exclude running/awaiting_hitl
  params.status = TERMINAL_STATUSES.join(",");
  // AND semantics: each active filter is passed independently
  if (outcomeFilter !== "all") params.outcome = outcomeFilter;
  if (topologyFilter !== "all") params.topology = topologyFilter;
  // Use 'repo' param for substring matching (supported by backend)
  if (repoFilter.trim()) params.repo = repoFilter.trim();

  const { data, isLoading, error } = useTasks({
    ...params,
    refetchInterval: 10_000, // VAL-HISTORY-012: new terminal task within 10s
  });

  const handleOutcomeChange = (value: string) => {
    updateParams({ outcome: value !== "all" ? value : null, page: null });
    setPage(0);
  };

  const handleTopologyChange = (value: string) => {
    updateParams({ topology: value !== "all" ? value : null, page: null });
    setPage(0);
  };

  const handleRepoChange = (value: string) => {
    updateParams({ repo: value || null, page: null });
    setPage(0);
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
    updateParams({ page: newPage > 0 ? String(newPage) : null });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Task History</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Filters — URL-preserved with AND semantics */}
        <div className="flex flex-wrap gap-3">
          <Select value={outcomeFilter} onValueChange={handleOutcomeChange}>
            <SelectTrigger className="w-[200px]">
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

          <Select value={topologyFilter} onValueChange={handleTopologyChange}>
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
            onChange={(e) => handleRepoChange(e.target.value)}
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

        {/* Empty state — VAL-HISTORY-007 */}
        {!isLoading && !error && !data?.tasks.length && (
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
                  <TableHead>ID</TableHead>
                  <TableHead>Repo</TableHead>
                  <TableHead>Issue</TableHead>
                  <TableHead>Topology</TableHead>
                  <TableHead>Outcome</TableHead>
                  <TableHead>Cost</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Ended</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.tasks.map((task: TaskListItem) => (
                  <TableRow key={task.id}>
                    <TableCell className="font-mono text-xs">
                      {shortId(task.id)}
                    </TableCell>
                    <TableCell className="font-medium">
                      <Link
                        href={`/tasks/${task.id}`}
                        className="text-primary hover:underline"
                      >
                        {task.repo_url.replace("https://github.com/", "")}
                      </Link>
                    </TableCell>
                    <TableCell>#{task.issue_number ?? "—"}</TableCell>
                    <TableCell className="text-xs">{task.topology}</TableCell>
                    <TableCell>
                      {task.outcome ? (
                        <Badge
                          variant="outline"
                          className={outcomeColors[task.outcome] ?? ""}
                        >
                          {task.outcome}
                        </Badge>
                      ) : (
                        <Badge
                          variant="outline"
                          className={statusColors[task.status] ?? ""}
                        >
                          {task.status}
                        </Badge>
                      )}
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

            {/* Pagination — VAL-HISTORY-002 */}
            <div className="flex items-center justify-between pt-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 0}
                onClick={() => handlePageChange(page - 1)}
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
                onClick={() => handlePageChange(page + 1)}
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
